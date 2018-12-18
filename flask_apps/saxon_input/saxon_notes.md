

TODO


1. get things setup on the white computer to run without me having to upload a new executable.
2. generating a pdf with the questions and thoughts from the kids and their scripture study.


dashboards...I think I need to split the dashboards up afterall. I need to be able to specify the book for the problems missed one.

main menu needs some work if not admin.

need to put the Child's name into the upper right hand corner of the nav bar.
implement a login for Kay

the login feature is bad. If i'm the last to login the kids could access my portal by simply going to the main_menu using the url and not the UI

use the production server
something I think I'll do is run the data when the script runs and keep it in memory
* use the login name and load the data and keep in memory at login.

todo: predict whether a child is going to miss the next problem from this chapter: condition on history with those types of problems, recent history, etc.
todo: click to zoom into seven or so days before and after that point.
-how do you have a click feature anywhere on the plot and not on a path or circle, etc.?
todo: make sure everything is correct

BACKEND
* move the functions to different scripts...saxon_math_command.py is getting very long...I can probably delete the other dashboards as well...don't need them now.
* use a gunicorn server. See server.py in flask_apps > model_app. I couldn't get this to work because I couldn't kill the gunicorn server. 
    * maybe this is helpful: https://gist.github.com/TheWaWaR/10955091?

TIME
* click on book and shows the datapoints, stats, and lines for that book
* be able to click and then get the stats only for that segment.

MIXED
* put a feature in that allows me to order bars by either mean or chapter origin number
* zoom in and out of supplementary plot.

TESTS
* red line that denotes the average across tests completed

VOCAB
* vocab dashboards
    * how many minutes per day
    * how many cards per day and of which type?
    * time per card
    * which cards in a chapter did he spend the most time on?
    * total time per chapter
    * total time per card in chapter

SCRIPTURE THOUGHTS

MAIN_MENU
* rows with headings



MAKE THE CHAPTER COME UP AUTOMATICALLY AND CHECK IF ALREADY BEEN ENTERED.




todo: make it so calvin can't go directory to the data entry pages
