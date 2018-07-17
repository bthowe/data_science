from jinja2 import Template
with open('email_template.html') as file_:
    template = Template(file_.read())
    t = template.render(
        hero_image='https://media.ldscdn.org/images/media-library/primary/childrens-songbook-illustrations/children-songbook-art-153084-gallery.jpg',
        missionary_message='''
        <p style="margin: 0; padding-top: .5cm">Hello Kansas City 3rd Ward!</p>
        <p style="...">We loved seeing everyone at church this week and enjoyed the new faces that could also come and the love that this ward shows to them! This week we hope all of you have been seeing the amazing blessings that can come from lighting the world this Christmas season!</p>
        <p style="margin: 0; padding-top: .5cm">In Helaman 14, Samuel the Lamanite says, "...for behold, there shall be great lights in heaven, insomuch that in the night before he cometh there shall be no darkness, insomuch that it shall appear unto man as if it was day."</p>
        <p style="margin: 0; padding-top: .5cm">The coming of Christ is always light and as we try and bring Christ into our lives more and the lives of people around us more we will also see that light of Christ in our lives!</p>
        ''',
        scripture_button='https://www.lds.org/scriptures/bofm/hel/14.3',
        scripture_prompt='Read Helaman 14',
        announcements='''
        <ul>
        <li>Rodrigo Cuellar and Darlene Wright are to be baptized this coming Saturday (12/23). The service will begin at 10:30 a.m. in the Relief Society room and should conclude around 11. Everyone is invited! </li>
        <li>ICYMI: Amanda Nelson was baptized yesterday and confirmed today in Sacrament Meeting. We are excited to have Amanda in our ward!</li>
        </ul>
        ''',
        dinners='''
        <li>12/18: Hepworth</li>
        <li>12/19: </li>
        <li>12/20: Semones</li>
        <li>12/21: McCain</li>
        <li>12/22: Daniels</li>
        <li>12/23: Kent</li>
        <li>12/24: Peters</li>
        ''',
        teamups='''
        <li>12/19: Lonny Kintner</li>
        <li>12/20: Tim Renshaw</li>
        <li>12/21: Bruce McCain</li>
        ''',
        missionary1='''
        <a href="https://www.facebook.com/keaton.brown.564?fref=ts">Elder Brown</a>
        ''',
        missionary2='''
        <a href="https://www.facebook.com/austin.denney.142">Elder Denney</a>
        '''

    )

Html_file = open('email.html', 'w')
Html_file.write(t)
Html_file.close()


# in browser url bar type
#   data:text/html, <html stuff copied from screen from print statement>
