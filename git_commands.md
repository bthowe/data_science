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
    


