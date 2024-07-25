import warnings
# warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
# warnings.filterwarnings("ignore", ".*divide by zero.*")
warnings.filterwarnings("ignore", ".*torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*")
warnings.filterwarnings("ignore", ".*`do_sample` is set to.*")
warnings.filterwarnings("ignore", ".*None of the inputs have requires_grad=True. Gradients will be None.*")

warnings.filterwarnings("ignore", ".*ou have modified the pretrained model configuration to control generation.*")
warnings.filterwarnings("ignore", ".*The input hidden states seems to be silently casted in float32.*")
