from jinja2 import Template
with open('template.html') as file_:
    template = Template(file_.read())
print(template.render(something='Great things are happening!', ayo='Ayo!'))

# in browser url bar type
#   data:text/html, <html stuff copied from screen from print statement>
