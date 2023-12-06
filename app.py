import os
from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_url_path='/static')
# 登录密钥
app.secret_key = os.urandom(24)

# 上传文件保存的目录
UPLOAD_FOLDER = 'static/images'
# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载预训练的MNIST模型
model = tf.keras.models.load_model('mnist_model.h5')


def allowed_file(filename):
    """检查上传的文件是否符合要求"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# python web(Flask)网页框架
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    username = user['username']

    user_images = [image for image in os.listdir(app.config['UPLOAD_FOLDER']) if image.startswith(username)]
    predicted_numbers = {image: None for image in user_images}
    return render_template('index.html', images=user_images, predicted_numbers=predicted_numbers)

# 上传
@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return redirect(request.url)

    user = session['user']
    username = user['username']

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
# 上传照片
    if file and allowed_file(file.filename):
        filename = username + '_' + file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))

    return "Invalid file."

# 删除照片
@app.route('/delete/<filename>')
def delete(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    username = user['username']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path) and filename.startswith(username):
        os.remove(file_path)
    return redirect(url_for('index'))

# 标定照片
@app.route('/annotate/<filename>')
def annotate(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    username = user['username']

    if filename.startswith(username):
        # 在此处添加图片标定的逻辑
        return "Annotating image: {}".format(filename)

    return redirect(url_for('index'))


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'user' not in session:
        return redirect(url_for('login'))

    selected_images = request.form.getlist('selected_images')
    predicted_numbers = {}

    for image in selected_images:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image)

        # 读取用户选择的图像文件
        image = Image.open(image_path)

        # 对图像进行预处理
        image = image.convert('L')
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = np.reshape(image, (1, 28, 28, 1))

        # 使用模型进行图像分类
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions)
        import uuid
        image_id = str(uuid.uuid4())
        predicted_numbers[image_id] = predicted_label

    user = session['user']
    username = user['username']
    user_images = [image for image in os.listdir(app.config['UPLOAD_FOLDER']) if image.startswith(username)]

    return render_template('index.html', images=user_images, predicted_numbers=predicted_numbers)


# 用户管理功能
users = [
    {'username': 'admin1', 'password': 'admin123'},
    {'username': 'user1', 'password': 'user123'},
    {'username': 'user2', 'password': 'user123'}
]


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not user_exists(username):
            create_new_user(username, password)
            return redirect('/login')
        else:
            error = 'User already exists'
            return render_template('register.html', error=error)
    else:
        return render_template('register.html', error='')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        for user in users:
            if user['username'] == username and user['password'] == password:
                session['user'] = user

                # 添加以下代码段
                user_images = [image for image in os.listdir(app.config['UPLOAD_FOLDER']) if image.startswith(username)]
                predicted_numbers = {image: None for image in user_images}

                return render_template('index.html', images=user_images, predicted_numbers=predicted_numbers)

        return render_template('login.html', message='Invalid username or password')

    return render_template('login.html', register_url=url_for('register'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/users')
def user_list():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('users.html', users=users)


@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not user_exists(username):
            create_new_user(username, password)
            return redirect('/users')
        else:
            error = 'User already exists'
            return render_template('create_user.html', error=error)
    else:
        return render_template('create_user.html', error='')


@app.route('/edit_user/<username>', methods=['GET', 'POST'])
def edit_user(username):
    if 'user' not in session:
        return redirect(url_for('login'))

    user = get_user(username)
    if user:
        if request.method == 'POST':
            new_username = request.form['username']
            new_password = request.form['password']
            edit_existing_user(username, new_username, new_password)
            return redirect('/users')
        else:
            return render_template('edit_user.html', user=user)
    else:
        return redirect('/users')


@app.route('/delete_user/<username>', methods=['GET'])
def delete_user(username):
    if 'user' not in session:
        return redirect(url_for('login'))

    if user_exists(username):
        delete_existing_user(username)
    return redirect('/users')


def user_exists(username):
    for user in users:
        if user['username'] == username:
            return True
    return False


def create_new_user(username, password):
    users.append({'username': username, 'password': password})


def get_user(username):
    for user in users:
        if user['username'] == username:
            return user
    return None


def edit_existing_user(username, new_username, new_password):
    for user in users:
        if user['username'] == username:
            user['username'] = new_username
            user['password'] = new_password


def delete_existing_user(username):
    for user in users:
        if user['username'] == username:
            users.remove(user)
            break


@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        batch_size = int(request.form['batch_size'])
        epochs = int(request.form['epochs'])
        learning_rate = float(request.form['learning_rate'])

        # 在这里执行模型训练的逻辑，使用上述参数进行训练

        return "Model training started with parameters:<br>Batch Size: {}<br>Epochs: {}<br>Learning Rate: {}".format(
            batch_size, epochs, learning_rate)

    return render_template('train.html')


if __name__ == '__main__':
    static_folder = os.path.join(app.root_path, 'static')
    app.static_folder = static_folder
    app.run(debug=True)
