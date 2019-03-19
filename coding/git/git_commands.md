```git
git status
```
* displays a list of files that have been modified since the last time changes were saved
* shows what is in staging
* shows what has been modified but not put in staging
```git
git diff
```
* compare the file as it currently is to what you last saved

```git
git add file_name
```
* adds a file to staging

```git
git reset
```
* removes files from the about to be commited area (use after a git add .)

```git
git diff -r HEAD path/to/file
```
* -r means compare to a particular version
* HEAD is a shortcut meaning most recent commit
* how files differ from the last saved revision

```git
git command -m "some message"
```
* saves the changes in the staging area

```git
git log
```
* view log of the project's history
* most recent entries are shown first
* if you want the most recent log use the -1 option

```git
git log filename
```
* show history of changes made to that file, rather than the entire repo

```git
git show <first few characters of a commit's hash>
```
* To view the details of a specific commit
```git
git show HEAD~1 
```
* shows the penultimate commit
* HEAD~2 would show the ultapenultimate commit

```git
git annote filename
```
* shows who made the last change to each line of a file and when

```git
git diff HEAD..HEAD~2
```
* shows the difference between the current and ultapenultimate commit

```git
git clean -n
```
* will show you a list of files that are in the repository, but whose history Git is not currently tracking
```git
git clean -f
```
* will remove these files
* be very careful with this...these are untracked files and so have not been saved

```git
git config --list 
```
* options include --system, --global, --local
* should change name and email address on every computer I use (i.e., use the global option): these keys are given by user.name and user.email, respectively
    * e.g., ```git config --global user.email my_email@gmail.com```
    
```git
git checkout -- filename
```
* will discard the changes that have not yet been staged in filename

```git
git reset HEAD filename
```
* will undo changes that have been staged in filename
* resets the file to the state last staged
* if you want to go all the way back to where you were before you started making changes, you must also do git checkout -- filename

```git
git checkout <hash> <name of file>
git checkout 2242bd report.txt
```
* restores old version of a file
* doesn't erase any of repo's history...restoring is another commit
* can use folder names instead of filenames (e.g., . to mean current directory)

```git
git diff branch1..branch2
```
* see differences between branches

```git
git checkout branch-name
```
* switch to another branch

```git
git checkout -b branch-name
```
* creates a new branch

```git
git merge source destination
```
* incorporate the changes made in source into destination

```git
git init project-name
```
* create a new repository

```git
git init file:///Users/travis.howe/repo new_name
```
* can also run git init on an existing repo

```git
git remote -v
```
* lists the origins

```git
git remote add remote-name URL
git remote rm remote-name
```
* add and remove existing remotes

```git
git pull origin branch_name
```
* pull changes

```git
git push origin branch_name
```
* push changes to remote

```git
git checkout branch1
git rebase master
```
Similar to merge, but fundamentally different. 
This moves the entire branch1 branch to begin on the tip of the master branch, effectively incorporating all of the new commits in master. 
But, instead of using a merge commit, rebasing re-writes the project history by creating brand new commits for each commit in the original branch.

```git
git cherry -v
```
* list the commits waiting to be pushed

```git
git checkout -b <new_branch_name>
git pull origin <new_branch_name>
```
* create branch new_branch_name locally and then pull from remote

```git
git fetch
```
git pull does a git fetch followed by a git merge