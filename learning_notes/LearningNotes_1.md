# 学习笔记-过程记录1

在学习过程中，常常用到命令行，尤其是在Linux shell和git中，因此我委托AI做了一个总结，将我在本项目中遇到的、使用到的命令行进行汇总。我为什么选择用AI呢？原因有二：其一是我的水平还达不到自己总结的地步，而且我的时间有限，我无法再如此之短的时间内完成这样精美的总结。其二就是，AI按照我的想法进行了总结，我看着感官还不错，因此我觉得作为一个工具，使用AI总结并不不妥。

# Dian 项目命令行速查表

> 命令的一般形式：
>
> `command [options] [arguments]`
>
> - `command`：要执行的命令/程序
> - `options`：控制命令如何执行
> - `arguments`：命令操作的对象
>
> 本表只整理本次 Dian 项目中实际接触和使用过的命令。

---

## 1. Linux / WSL：目录与文件操作

> 使用场景：Ubuntu / WSL 的 Shell 中操作文件、目录。
>
> 其中大部分是 Linux Shell 命令，不要默认认为 Windows PowerShell 也完全相同。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `pwd` | 显示当前所在目录 | `pwd` |
| `cd` | 切换目录 | `cd ~/projects/dian-algorithm` |
| `cd ..` | 回到父目录 | `cd ..` |
| `cd ~` | 回到当前用户 Home | `cd ~` |
| `ls` | 查看目录内容 | `ls` |
| `ls -l` | 以详细格式查看 | `ls -l` |
| `ls -a` | 显示全部，包括隐藏文件 | `ls -a` |
| `ls -la` | 全部文件 + 详细信息 | `ls -la` |
| `ls -lh` | 详细显示，文件大小更易读 | `ls -lh model.pth` |
| `mkdir` | 创建目录 | `mkdir level4_unet` |
| `mkdir -p` | 连同不存在的父目录一起创建 | `mkdir -p outputs/final_test` |
| `touch` | 创建空文件/更新时间戳 | `touch README.md` |
| `cat` | 直接输出文件内容 | `cat .gitignore` |
| `less` | 分页查看长文本 | `less README.md` |
| `head` | 查看文件开头 | `head README.md` |
| `tail` | 查看文件末尾 | `tail README.md` |
| `tail -n` | 查看最后指定行数 | `tail -n 30 README.md` |
| `cp` | 复制文件 | `cp source.txt target.txt` |
| `cp -r` | 递归复制目录 | `cp -r source_dir target_dir` |
| `mv` | 移动或重命名 | `mv old.md README.md` |
| `rm` | 删除文件 | `rm temp.txt` |
| `rm -r` | 递归删除目录 | `rm -r temp_dir` |
| `rm -rf` | 强制递归删除，慎用 | `rm -rf temp_dir` |
| `which` | 查看实际调用的程序位置 | `which python` |

---

## 2. Linux / WSL：查找与处理文本

> 使用场景：检查项目文件、模型文件、README、Python import 等。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `find` | 递归查找文件/目录 | `find level4_unet` |
| `find -type f` | 只查普通文件 | `find level4_unet -type f` |
| `find -name` | 按名称匹配 | `find level4_unet -name "*.pth"` |
| `find -maxdepth` | 限制向下搜索层数 | `find level4_unet -maxdepth 3` |
| `find -size` | 按文件大小筛选 | `find level4_unet -type f -size +20M` |
| `find -exec` | 对找到的文件执行命令 | `find level4_unet -name "*.pth" -exec ls -lh {} \;` |
| `find -printf` | 按指定格式输出结果 | `find level4_unet -type f -printf '%p\n'` |
| `grep` | 查找/筛选匹配文本 | `grep '\.pth$'` |
| `grep -E` | 使用扩展正则表达式 | `grep -E '__pycache__\|\.pyc$'` |
| `grep -h` | 匹配多个文件时不显示文件名 | `grep -h "import" *.py` |
| `grep -hE` | `-h` 与 `-E` 组合 | `grep -hE '^(import\|from) ' *.py` |
| `sort` | 对文本行排序 | `sort file.txt` |
| `sort -u` | 排序并去重 | `sort -u` |
| `wc -l` | 统计行数 | `wc -l` |
| `sed -n` | 只输出指定内容，不默认全输出 | `sed -n '1,80p' README.md` |

---

## 3. Shell 语法：路径、组合与重定向

> 使用场景：Linux / WSL Shell。
>
> 这一类很多并不是独立 command，而是 Shell 本身的语法。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `/` | Linux 根目录；也用于分隔路径 | `/home/wangzehao` |
| `~` | 当前用户 Home 目录 | `cd ~` |
| `.` | 当前目录 | `python ./train.py` |
| `..` | 父目录 | `cd ..` |
| `*` | Shell 通配符，匹配任意字符 | `level4_unet/*.py` |
| `$变量名` | 读取 Shell 变量 | `echo $PATH` |
| `>` | 将输出覆盖写入文件 | `echo "hello" > test.txt` |
| `>>` | 将输出追加到文件末尾 | `echo "hello" >> test.txt` |
| `\|` | 管道：左侧输出交给右侧输入 | `git status --short \| tail -n 30` |
| `&&` | 左侧成功后才执行右侧 | `mkdir test && cd test` |
| `\|\|` | 左侧失败后执行右侧 | `grep '\.pth$' \|\| true` |
| `<<'EOF'` | 输入多行文本，直到遇到结束标记 | `cat > file.txt <<'EOF' ... EOF` |
| `"..."` | 双引号，把含空格内容作为整体等 | `git commit -m "Complete Level 4"` |
| `'...'` | 单引号，尽量按字面保护其中字符 | `grep '\.pth$'` |
| `\;` | 结束 `find -exec` 后的命令 | `-exec ls -lh {} \;` |
| `{}` | `find -exec` 中代表当前找到的文件 | `-exec ls -lh {} \;` |

---

## 4. WSL / GPU 环境检查

> 使用场景：Windows + WSL 开发环境配置、确认 GPU 是否正常。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `wsl` | 从 Windows 进入/调用 WSL | `wsl` |
| `nvidia-smi` | 查看 NVIDIA GPU、驱动及 GPU 状态 | `nvidia-smi` |

### Windows 与 WSL 路径对应

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `/mnt/c/` | WSL 中访问 Windows C 盘 | `/mnt/c/Users/...` |
| `/mnt/d/` | WSL 中访问 Windows D 盘 | `/mnt/d/User/University/QQ/Dian/deli` |

---

## 5. Conda：Python 环境管理

> 使用场景：为 Dian 项目建立独立 Python 环境。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `conda create` | 创建 Conda 环境 | `conda create -n dian-ai python=3.11.9` |
| `conda create -n` | `-n` 指定环境名称 | `conda create -n dian-ai python=3.11.9` |
| `conda activate` | 激活环境 | `conda activate dian-ai` |
| `conda deactivate` | 退出当前环境 | `conda deactivate` |
| `conda env list` | 查看已有环境 | `conda env list` |
| `conda info --envs` | 查看已有 Conda 环境 | `conda info --envs` |
| `conda list` | 查看当前环境安装的包 | `conda list` |

---

## 6. Python / pip：运行程序与管理依赖

> 使用场景：运行训练、测试、推理程序，以及检查/安装 Python 包。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `python` | 使用当前 Python 解释器运行程序 | `python train.py` |
| `python --version` | 查看 Python 版本 | `python --version` |
| `python -c` | 直接执行一小段 Python 代码 | `python -c "import torch; print(torch.__version__)"` |
| `python -` | 从标准输入读取 Python 代码 | `python - <<'PY' ... PY` |
| `pip install` | 安装 Python 包 | `pip install matplotlib` |
| `pip install -r` | 按 requirements 文件安装依赖 | `pip install -r requirements.txt` |
| `pip list` | 查看已安装 Python 包 | `pip list` |
| `pip show` | 查看某个包的信息 | `pip show torch` |

---

## 7. Git：仓库与状态检查

> 使用场景：本地版本控制。
>
> Git ≠ GitHub。这里的命令首先操作的是本地 Git 仓库。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `git init` | 初始化 Git 仓库 | `git init` |
| `git status` | 查看当前仓库状态 | `git status` |
| `git status --short` | 简洁显示仓库状态 | `git status --short` |
| `git branch --show-current` | 查看当前分支 | `git branch --show-current` |
| `git log` | 查看提交历史 | `git log` |
| `git log --oneline` | 一行显示一个 commit | `git log --oneline` |
| `git log -5` | 只看最近 5 个 commit | `git log -5` |
| `git --no-pager` | 输出时禁用分页器 | `git --no-pager log --oneline -5` |

---

## 8. Git：add / diff / commit

> 使用场景：
>
> `Working Directory → Staging Area → Local Repository`

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `git add` | 把修改加入暂存区 | `git add level4_unet` |
| `git diff` | 查看尚未 staged 的变化 | `git diff` |
| `git diff --cached` | 查看已经 staged 的变化 | `git diff --cached` |
| `git diff --cached --stat` | 统计即将提交的文件变化 | `git diff --cached --stat` |
| `git diff --cached --name-only` | 只列出即将提交的文件名 | `git diff --cached --name-only` |
| `git commit` | 将暂存区内容提交到本地仓库 | `git commit -m "Complete Level 4"` |
| `git commit -m` | `-m` 直接指定 commit 信息 | `git commit -m "Complete Level 4"` |

---

## 9. Git：`.gitignore` 检查

> 使用场景：控制哪些数据集、缓存、checkpoint 不进入 Git。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `.gitignore` | Git 忽略规则文件 | `level1_mlp/data/` |
| `*` | 在 ignore 规则中用于通配 | `*.pyc` |
| `**` | 匹配多层目录 | `outputs/**/*.pth` |
| `!` | 取消前面的忽略规则 | `!outputs/train_weighted_loss/best_unet_weighted.pth` |
| `git check-ignore` | 检查文件是否被 ignore | `git check-ignore model.pth` |
| `git check-ignore -v` | 同时显示匹配了哪条 ignore 规则 | `git check-ignore -v model.pth` |

---

## 10. Git：用户信息

> 使用场景：设置 Git commit 的作者信息。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `git config` | 查看/修改 Git 配置 | `git config user.name` |
| `git config --global` | 修改当前用户的全局 Git 配置 | `git config --global user.name "Austin-W123"` |
| `user.name` | Git commit 作者名称 | `git config --global user.name "Austin-W123"` |
| `user.email` | Git commit 作者邮箱 | `git config --global user.email "..."` |

---

## 11. Git + GitHub：远程仓库

> 使用场景：把本地 Git 仓库连接到 GitHub。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `git remote` | 管理远程仓库 | `git remote -v` |
| `git remote -v` | 查看远程仓库及地址 | `git remote -v` |
| `git remote add` | 添加一个远程仓库 | `git remote add origin <SSH地址>` |
| `origin` | 我们给 GitHub 远程仓库使用的名称 | `git push origin main` |
| `main` | 本项目使用的主分支名称 | `git push origin main` |
| `git push` | 将本地 commit 推送到远程仓库 | `git push origin main` |
| `git push origin main` | 将本地 `main` 推送到 `origin` | `git push origin main` |

### 本次 Git 数据流

`修改文件 → git add → git commit → git push origin main → GitHub`

---

## 12. SSH：GitHub 身份认证

> 使用场景：让本机通过 SSH 安全连接 GitHub，从而执行 Git push。

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `ssh-keygen` | 生成 SSH 密钥对 | `ssh-keygen -t ed25519 -C "..."` |
| `ssh-keygen -t` | 指定密钥算法类型 | `ssh-keygen -t ed25519` |
| `ssh-keygen -C` | 给密钥添加注释 | `ssh-keygen -C "..."` |
| `ssh -T` | 测试 SSH 身份认证 | `ssh -T git@github.com` |
| `id_ed25519` | SSH 私钥，不能公开 | `~/.ssh/id_ed25519` |
| `id_ed25519.pub` | SSH 公钥，可添加到 GitHub | `~/.ssh/id_ed25519.pub` |
| `chmod` | 修改 Linux 文件权限 | `chmod 600 ~/.ssh/id_ed25519` |
| `chmod 600` | 仅文件所有者可读写 | `chmod 600 ~/.ssh/id_ed25519` |

---

## 13. 本项目值得记住的组合命令

| 字符 | 含义 | 简单例子 |
|---|---|---|
| `find ... -exec ...` | 找到文件后逐个执行命令 | `find level4_unet -type f -name "*.pth" -exec ls -lh {} \;` |
| `git ... \| grep ...` | Git 输出继续交给 grep 筛选 | `git --no-pager diff --cached --name-only \| grep '\.pth$'` |
| `... \| wc -l` | 对上一命令输出的行数计数 | `git diff --cached --name-only \| wc -l` |
| `grep ... \| sort -u` | 查找后排序并去重 | `grep -hE '^(import\|from) ' level4_unet/*.py \| sort -u` |
| `grep ... \|\| true` | 即使 grep 没找到也保持命令链成功 | `grep '\.pth$' \|\| true` |

---

# 最核心的记忆链

## Linux / Shell

`pwd → cd → ls → mkdir → cat → find → grep → |`

## Conda / Python

`conda create → conda activate → python → pip`

## Git

`git status → git add → git diff --cached → git commit → git log`

## GitHub

`git remote → SSH → git push origin main`

## Git 四个位置

`Working Directory → Staging Area → Local Repository → Remote Repository`

对应：

`修改 → git add → git commit → git push`