import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

#======================= yaha hm UI bnaenge and UI me jo hm model trained kiye hai usko call kr lenge===============

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# class to used upload form

class UploadForm(FlaskForm):
    content = FileField('Content Image') # filed hoga UI ka
    style = FileField('Style Image') # filed hoga UI ka
    content_path = HiddenField()  # path directory
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

# yaha hm model load kr rahe hai
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yaha hm model encoder me pass kr rahe hai
encoder = VGGEncoder('vgg_normalised.pth').to(device)

# yaha decoder me pass kr rahe hai
decoder = Decoder().to(device)


# Load decoder model
decoder_path = r"E:\AI Major Project\Practice\ai-nst-project-main\ai-nst-project-main\NST_Code\experiment\exp1\decoder_20.pth"

decoder.load_state_dict(torch.load(decoder_path, map_location=device))

# decoder load kr rahe hai
# decoder.load_state_dict(torch.load('E:/AI Major Project/Practice/ai-nst-project-main/ai-nst-project-main/NST_Code/experiment/final_exp/decoder_final.pth'))


# decoder.load_state_dict(torch.load('E:\AI Major Project\Practice\ai-nst-project-main\ai-nst-project-main\NST_Code\experiment\exp1\decoder_20.pth'))
# yaha hme model ko train nii krna hai bs used krni hai,eval krni hai
encoder.eval()
decoder.eval()

# jo file UI se load hui hai wo allowed (jpg,png)hai ki nii model k liye iske liye ye fun hai
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# yaha hm style image ko convert kr rahe hai
def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
# hm content image ko transform kr rahe hai
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    #  yaha finally transform kr lenge
    content_image = content_transform(content_image).unsqueeze(0).to(device) # yaha hm content image tensor me transform kr rahe hai
    style_image = style_transform(style_image).unsqueeze(0).to(device) # yaha hm style image tensor me transform kr rahe hai
# unsqueeze krk 4D me convert krte hai,batchsize bhi dal rahe hai
# yaha hm transform kiye hai unko pass kr denge model ko
# 
    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True) # hm last wala feature Adain me pass krnge is liye test=true kiye hai
        style_feats = encoder(style_image, is_test=True)

# yaha hm AdaIN layer ko content_feats ,style_feats pass kr denge

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)


        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

# yaha hm final style=generated image ko return kr rahe hai
    return stylized_image

# basically decoder o/p image =tensor hai so cpu pr move krk
def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)  # squeeze krk 3D me convert krte hai,batch size ko hta denge
    image = image.clamp(0, 1)  # model o/p-0,1 se bahar hai to clamp krte hai taki 0,1 me laega
    image = transforms.ToPILImage()(image) # ab PI image me tranform kr lenge
    image.save(path)


# app ka main route yaha de rahe hai,yaha get post impl kr rahe hai
# define kr rahe hai index
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()  # form create kr rahe hai
    # yaha result k liye 
    result_image = None
    content_filename = None
    style_filename = None
    error = None

# yaha hm form ko validate kr rahe hai
    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:   # agar content data  hai & filename bhi hai
            if allowed_file(form.content.data.filename): # file jo upload hua hai wo allowed hai ya ni yaha check kr rahe hai
                # hai to aage badhte hai
                content_filename = secure_filename(form.content.data.filename)
                # yaha save kr dete hai us location pr
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
                # is path pr save kr rahe hai=content_path
        else:
            # agar content data nii hai
            content_filename = form.content_path.data
# agar filename nii hai to update kr denge upload_folder ko
        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename  # yaha filename upadte ho jaega
        else:
            style_filename = form.style_path.data
# yaha hm content and style dono check kr rahe hai dono image(c,s) ko load kr rahe hai
        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                # yaha hm dono style and content image ko convert kr denge 3 d me
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
    else:
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# examples k undar bhi koi file ya directory hoti hai usko bhi acces kr paega
@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

# yaha run krnge localhost me
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)






