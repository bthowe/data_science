from jinja2 import Template
with open('email_template.html') as file_:
    template = Template(file_.read())
    t = template.render(
        # hero_image="https://media.ldscdn.org/images/media-library/gospel-art/book-of-mormon/and-he-healed-them-all-every-one-290248-print-do-not-copy-notice.jpg",
        hero_image="https://knowhy.bookofmormoncentral.org/sites/default/files/knowhy-img/2016/3/main/doctrineofchrist.jpg",
        missionary_message='''
        <p style="margin: 0; padding-top: .5cm">Greetings Kansas City 3rd Ward!</p>
        <p>“For I neither received it of man, neither was I taught it, but by the revelation of Jesus Christ." (Galatians 1: 12) These are the words of Paul, a powerful disciple of Christ in the New Testament, while he was telling his story of conversion to the Galatians who had only ever heard myths about him before. This dictates clearly the way that we receive our testimony of our Lord Jesus Christ, through the Holy Ghost and revelation.</p>
        <p>Let us all hearken unto the words of our Prophet, President Nelson, and apply his advice given to us on receiving personal revelation in this past General Conference and specifically apply it to our own personal testimonies.</p>
        <p>We promise that as you study the word of God and ponder on what he would like you to do next, that you will receive the guidance necessary to grow your testimonies and become worthy disciples of Jesus Christ.</p>
        <p>We love you and pray for you!</p>
        ''',
        scripture_button='https://www.lds.org/scriptures/nt/gal/1.12?lang=eng',
        scripture_prompt="Read Galatians 1",
        tip_of_week='''
        <p>We succeed as member missionaries when we invite people to learn and accept the truth regardless of how that invitation is received.</p>
        ''',
        tip_button="https://www.lds.org/ensign/2005/02/seven-lessons-on-sharing-the-gospel?lang=eng#title5",
        tip_prompt="Read Ensign Article",
        announcements='''
        <ul>
        <li>ICYMI: Chris Lawrie was baptized and confirmed a member of the Church this past weekend. Congratulations to him and his family!</li>
        </ul>
        ''',
        dinners='''
        <li>9/10: Pace</li>
        <li>9/11: Allen</li>
        <li>9/12: Davis</li>
        <li>9/13: Ellis</li>
        <li>9/14: Davidson</li>
        <li>9/15: Crawford</li>
        <li>9/16: Bakkedahl</li>
        ''',
        teamups1='''
        <li>9/11: Russell Hayes</li>
        <li>9/12: Jacob Henderson</li>
        <li>9/13: Steve Hirschi</li>
        ''',
        teamups2='''
        <li>9/18: Brock Josephson</li>
        <li>9/19: Carl Kent </li>
        <li>9/20: Lonny Kintner</li>
        ''',
        teamups3='''
        <li>9/25: Terry Lewis</li>
        <li>9/26: David Martin</li>
        <li>9/27: Phillip Lunceford</li>
        ''',
        teamups4='''
        <li>10/2: Bruce McCain</li>
        <li>10/3: Robert Mayo</li>
        <li>10/4: Adam McDonald</li>
        ''',
        missionary1='''
        <a href="https://www.facebook.com/hunter.coley.96">Elder Coley</a>
        ''',
        missionary2='''
        <a href="https://www.facebook.com/isaac.watts.503">Elder Watts</a>
        '''
    )

Html_file = open('email.html', 'w')
Html_file.write(t)
Html_file.close()


# in browser url bar type
#   data:text/html, <html stuff copied from screen from print statement>
