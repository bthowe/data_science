from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_mongoalchemy import MongoAlchemy
from flask import Flask, render_template, redirect, url_for, request, session
from wtforms.validators import InputRequired, Email, Length
from wtforms import StringField, PasswordField, BooleanField
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret '
app.config['MONGOALCHEMY_DATABASE'] = 'library'
Bootstrap(app)

# db = MongoAlchemy(app)
client = MongoClient()
db = client['library']

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# class User(UserMixin, db.Document):
#     id = db.Document.Index()
#     username = db.StringField()
#     email = db.StringField()
#     password = db.StringField()


# class User():
#     def query(self):
#         pass
#     def document_create(self, username, email, password):
#         db['user'].insert_one({'username': username, 'email': email, 'password': password})


    # id = db.Document.Index()
    # username = db.StringField()
    # email = db.StringField()
    # password = db.StringField()

@login_manager.user_loader
def load_user(user_id):
    user = list(db['user'].find({'_id': user_id}))[0]
    print(user)
    if not user:
        return None
    return user['username']

    # return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email('Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter(User.username == form.username.data).first()
        print(user.password)
        print(user.username)
        print(user.email)
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
        return '<h1>Invalid username or password</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        User(username=form.username.data, email=form.email.data, password=hashed_password).save()
        return '<h1>New user has been created</h1>'

    return render_template('signup.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)