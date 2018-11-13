source ~/PycharmProjects/venv/data_science/bin/activate

cd ~/Projects/github/data_science/flask_apps/saxon_input

git pull origin master

mongod --fork --logpath ../../bash/vocab/mongo_log/mongod.log

mongorestore --db vocab --verbose dump/vocab
mongorestore --db math_book_info --verbose dump/math_book_info
mongorestore --db math_exercise_origins --verbose dump/math_exercise_origins
mongorestore --db math_performance --verbose dump/math_performance
mongorestore --db scripture_commentary --verbose dump/scripture_commentary

open -a "Google Chrome" http://0.0.0.0:8001/login
python3 saxon_math_command.py

mongodump -d math_book_info -o dump
mongodump -d math_exercise_origins -o dump
mongodump -d math_performance -o dump
mongodump -d vocab -o dump
mongodump -d scripture_commentary -o dump

git add dump/.
git commit -m "standard mongo dump"
git push origin master

killall mongod

deactivate