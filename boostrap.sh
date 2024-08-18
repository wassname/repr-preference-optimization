# init a runpod env
apt update
apt install -y tig vim htop nvtop tmux git curl wget unzip zip jq tree zsh zsh-antigen fizsh ncdu neovim sudo cron lsyncd ranger direnv python3-pygments powerline xsel xclip yadm fzf autojump fish fasd

# my dotfiles?
# # configure git email and cache... wait this might be in yadm
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=360000'
git config --global user.name "wassname"
git config --global user.email "github@wassname"
git config --global core.editor "vim"
git config --global pull.rebase false
git config --global push.default simple

pip install poetry
poetry install
poetry shell
python -m ipykernel install --user --name repr
pip install wheel
pip install flash-attn --no-build-isolation -q


yadm clone https://github.com/wassname/dotfiles.git
yadm status
yadm decrypt
yadm bootstrap

wandb login