#!/bin/bash
apt update
apt install -y tig vim htop nvtop ncdu ranger \
    tmux git curl wget unzip zip jq tree direnv \
    zsh zsh-antigen fizsh \
    python3-pygments powerline xsel xclip libaio-dev \
    neovim sudo cron lsyncd yadm \
    fzf autojump fish fasd exa bat ripgrep


# add just
wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | sudo tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null
echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr $(lsb_release -cs)" | sudo tee /etc/apt/sources.list.d/prebuilt-mpr.list
sudo apt update
sudo apt install just

# git config --global credential.helper 'cache --timeout=360000'
yadm clone https://github.com/wassname/dotfiles.git
# yadm checkout runpod
yadm stash
yadm status

# change shell to zsh
chsh -s $(which zsh)
zsh

cd /workspace/repr-preference-optimization
. ./.venv/bin/activate
wandb login
