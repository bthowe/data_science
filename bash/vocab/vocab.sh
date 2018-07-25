source ~/PycharmProjects/venv/data_science/bin/activate

cd ~/Projects/github/data_science/flask_apps/vocab_flashcards

mongod --fork --logpath ../../bash/vocab/mongo_log/mongod.log
python3 vocab_app.py
killall mongod

deactivate