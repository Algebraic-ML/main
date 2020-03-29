# 如何使用git来上传文件
## 一、配置本地git
1. 下载安装git和vscode
2. 在本地新建专属合作的文件夹
3. vscode打开该文件夹,在终端中初始化git. 使用`git init`创建本地仓库. 本地仓库的管理办法请参考 https://www.runoob.com/git/git-basic-operations.html. vscode里也有内置的**一键**git功能, 例如Commit(暂存更改)
4. 按照 https://www.jianshu.com/p/ad59186b6381 的方法, 使用ssh的方式链接本地仓库与云端仓库. 其中, 最后输入`git remote add origin git@github.com:Algebraic-ML/main.git`以关联本地文件夹和云端仓库

## 二、同步编辑
1. 在本地使用时,git使用分支的方法管理版本. 而在同步本地和云端内容时, 使用pull和push两个命令. 在vscode中也有内置的功能对应了拉取和推送. 
2. 首先, 使用`git pull <远程主机名> <远程分支名>:<本地分支名>`命令将远程仓库拉取至本地分支,并将它们合并. 通常使用`git pull origin master`将远程的master分支同步到本地. 此外, 使用`git fetch`和`git merge`两个命令也能达到同样效果. 
3. 当编辑完本地内容时, 使用`git push <远程主机名> <本地分支名>:<远程分支名>`命令可以推送本地分支到远程, 并将其合并. 通常使用`git push origin master`
4. 以上内容基本可以在vscode里面一键完成

## 三、分支管理
1. 主要针对本地版本管理. 
2. 当遇到大范围编辑修改或创建新版本时, 可以用`git branch <分支名>`创建新分支, 并输入`git checkout <分支名>`进入该分支. 使用`git branch`查看当前所有分支
3. `git merge <分支名>`可以把当前分支与目标分支合并