from eval import eval_tbnet 
import html
from flask import render_template, Flask
import os

app = Flask(__name__)
app.app_context().push()
# decorator to access the app
@app.route("/")
# @app.route("/index")

def home():
    # a template folder is necessary for render_template()
    string = eval_tbnet()
    path = "templates"
    # Check whether the templates path exists in the directory
    isExist = os.path.exists(path)
    
    if not isExist:
        os.makedirs(path)
    
    os.chdir(".\\templates\\")

    string = string.replace('\n', '<br>')
    # string = string.replace("<", "&lt;")
    # string = string.replace(">", "&gt;")

    with open("top.html", "w", encoding="utf-8") as file:
        file.write("<html>" + string + "</html>")

    return render_template("top.html")

if __name__ == '__main__':
    app.run()