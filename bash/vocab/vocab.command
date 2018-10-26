source ~/PycharmProjects/venv/data_science/bin/activate

cd ~/Projects/github/data_science/flask_apps/vocab_flashcards

mongod --fork --logpath ../../bash/vocab/mongo_log/mongod.log
open -a "Google Chrome" http://0.0.0.0:8001/
python3 vocab_app.py

mongodump -d vocab -o dump

git add dump/.
git commit -m "standard vocab mongo dump"
git push origin master

killall mongod

deactivate