import os
from interface import interface
from flask import render_template

@interface.route('/')
@interface.route('/index')
def index():
    images = []

    for r, d, f in os.walk('../data/16_0/'):
        for file in f:
            if ".png" in file:
                images.append(file)
                        
    print('\n'.join(images))
    return render_template('index.html', title='Home', images=sorted(images))