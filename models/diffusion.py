import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, c_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0.float())
        x_t = (
            extract(self.sqrt_alphas_bar.to(x_0.device), t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar.to(x_0.device), t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t, c_0), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, c):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, c)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t, c)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t, c)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, c_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        c_t = c_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, c=c_t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', eta=0.0): # eta for DDIM noise control
        super().__init__()
        self.model = model
        self.T = T
        self.img_size = img_size
        # DDIM typically works best when the model predicts epsilon (the noise)
        if mean_type != 'epsilon':
            print(f"Warning: DDIMSampler is generally designed for mean_type 'epsilon'. "
                  f"Current mean_type is '{mean_type}'. Ensure model output is handled correctly.")
        self.mean_type = mean_type
        self.eta = eta # eta = 0 for deterministic DDIM, eta = 1 for DDPM-like noise

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        
        # Precompute sqrt terms for DDIM update
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - self.alphas_bar))
        
        # For predicting x_0 from epsilon
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / self.alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / self.alphas_bar - 1))

    def predict_xstart_from_eps(self, x_t, t, eps):
        # Utility to predict x_0 from x_t and predicted noise eps
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def forward(self, x_T, c_T, num_steps=None, eta=None):
        """
        DDIM sampling process.
        Args:
            x_T (torch.Tensor): The initial noise tensor (e.g., from torch.randn_like).
            c_T (torch.Tensor): The conditional input.
            num_steps (int, optional): Number of DDIM steps. If None, defaults to self.T.
                                       For acceleration, use a value much smaller than self.T.
            eta (float, optional): Controls the stochasticity. 0 for deterministic DDIM.
                                   If None, uses self.eta from initialization.
        Returns:
            torch.Tensor: The generated sample x_0.
        """
        if num_steps is None:
            num_steps = self.T
        if eta is None:
            eta = self.eta

        img = x_T
        c_t = c_T # Assuming condition c does not change with t

        # Define the sequence of timesteps for DDIM sampling (from T-1 down to 0)
        # This creates `num_steps` timesteps, spaced out over the original T steps.
        times = torch.linspace(self.T - 1, 0, num_steps).long().to(x_T.device)

        for i in range(num_steps):
            t_val = times[i]
            t = torch.full((x_T.shape[0],), t_val, device=x_T.device, dtype=torch.long)

            # Get the previous timestep (t_prev). For the last step, t_prev is effectively -1.
            t_prev_val = times[i+1] if i < num_steps - 1 else -1
            t_prev = torch.full((x_T.shape[0],), t_prev_val, device=x_T.device, dtype=torch.long)

            # 1. Predict model output (epsilon or x_0)
            model_output = self.model(img, t, c_t)

            if self.mean_type == 'epsilon':
                eps = model_output
                # Predict x_0 using the model's noise prediction
                pred_x0 = self.predict_xstart_from_eps(img, t, eps)
            elif self.mean_type == 'xstart':
                pred_x0 = model_output
                # If model predicts x_0, derive eps for the DDIM formula
                # eps = (x_t - sqrt(alpha_bar_t)*x_0) / sqrt(1-alpha_bar_t)
                eps = (img - extract(self.sqrt_alphas_bar, t, img.shape) * pred_x0) / \
                      extract(self.sqrt_one_minus_alphas_bar, t, img.shape)
            else:
                raise NotImplementedError(f"DDIM for mean_type '{self.mean_type}' is not directly supported in this script. "
                                          "Please use 'epsilon' or 'xstart'.")

            # Clamp predicted x_0 to be within valid range (e.g., [-1, 1] if data is normalized)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # 2. Get alpha_bar_t and alpha_bar_t_prev
            alpha_bar_t = extract(self.alphas_bar, t, img.shape)
            # If t_prev is -1, alpha_bar_t_prev is 1 (corresponds to alpha_bar_0)
            alpha_bar_t_prev = extract(self.alphas_bar, t_prev, img.shape) if t_prev_val >= 0 else torch.ones_like(alpha_bar_t)

            # 3. Calculate DDIM update terms (Equation 12 from DDIM paper)
            # sigma_t = eta * sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            # Ensure variance term inside sqrt is non-negative
            variance_term_numerator = (1 - alpha_bar_t_prev)
            variance_term_denominator = (1 - alpha_bar_t)
            variance_ratio = (1 - alpha_bar_t / alpha_bar_t_prev) if t_prev_val >=0 else 0 # Avoid division by zero if alpha_bar_t_prev is 0
            
            # Handle potential division by zero or small numbers if alpha_bar_t is close to 1
            if isinstance(variance_term_denominator, torch.Tensor):
                variance_term_denominator = torch.where(variance_term_denominator == 0, torch.ones_like(variance_term_denominator)*1e-6, variance_term_denominator)
            else: # float
                variance_term_denominator = max(variance_term_denominator, 1e-6)


            variance = (variance_term_numerator / variance_term_denominator) * variance_ratio
            variance = torch.clamp(variance, min=1e-20) # Ensure non-negativity due to potential floating point issues
            sigma_t = eta * torch.sqrt(variance)

            # Equation for x_{t-1}
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + 
            #           sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps + 
            #           sigma_t * random_noise
            
            term1_coeff_sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_t_prev)
            
            term2_coeff_sqrt_term = 1. - alpha_bar_t_prev - sigma_t**2
            term2_coeff_sqrt_term = torch.clamp(term2_coeff_sqrt_term, min=0.0) # Clamp for numerical stability
            term2_coeff = torch.sqrt(term2_coeff_sqrt_term)

            # Generate random noise only if eta > 0 and not the last step
            random_noise = torch.randn_like(img) if t_prev_val > -1 and eta > 0 else torch.zeros_like(img)

            img = term1_coeff_sqrt_alpha_bar_prev * pred_x0 + \
                  term2_coeff * eps + \
                  sigma_t * random_noise
        
        x_0 = img
        return torch.clamp(x_0, -1.0, 1.0)