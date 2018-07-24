from jinja2 import Template
with open('email_template.html') as file_:
    template = Template(file_.read())
    t = template.render(
        hero_image='https://www.mormonnewsroom.org/media/960x540/Kansas-City-Missouri-Temple1.jpg',
        missionary_message='''
        <p style="margin: 0; padding-top: .5cm">Hello Kansas City 3rd Ward!</p>
        <p> In a talk about commitment given in the October 1983 General Conference, Elder Marvin Ashton said </p>
            <blockquote>
                True happiness is not made in getting something. True happiness is becoming something. This can be done by being committed to lofty goals. We cannot become something without commitment...Each day we must be committed to lofty Christian performance because commitment to the truths of the Gospel of Jesus Christ is essential to our eternal joy and happiness. The time to commit and recommit is now.
            </blockquote>
        <p style="...">Let us all recommit ourselves when we take the sacrament this coming Sunday to living the Gospel of Jesus Christ more fully. Let us think of a righteous goal for our lives and then take the steps required to accomplish it. We promise that as you prayerfully ponder, set goals, and strive your absolute best to complete them, God will help you every step along the way and you will look back and say that, through the process, you are a better person. There is no greater joy than righteous progression.</p>
        ''',
        scripture_button='https://www.lds.org/general-conference/1983/10/the-word-is-commitment?lang=eng',
        scripture_prompt="Read Elder Ashton's talk",
        announcements='''
        <ul>
        <li>Emily McCulloch is scheduled to be baptized this coming Saturday (7/28). The time is still TBD. </li>
        <li>Beginning the week of July 29th, members of the Elders Quorum will have periodic team-up assignments with the full-time missionaries. These typically take place in the evening (and last between one and two hours) but can occur in the morning or afternoon if that is better for your schedule.</li>
        <li>Finally, over the next few weeks (before summer vacation ends) the full-time missionaries will begin visiting and teaching lessons to the young men and young women of the ward. Please welcome them into your home as they try to strengthen our youth and their friends.</li>


        </ul>
        ''',
        dinners='''
        <li>7/23: Ellibee</li>
        <li>7/24: Bakkedahl</li>
        <li>7/25: </li>
        <li>7/26: McCain</li>
        <li>7/27: Hornaday</li>
        <li>7/28: Gammon</li>
        <li>7/29: Jenkins</li>
        ''',
        teamups='''
        <li>7/31: James Allen</li>
        <li>8/1: Andy Bakkedahl</li>
        <li>8/2: Nathan Breneman</li>
        <li>8/7: Sam Lunceford</li>
        <li>8/8: Will Burke</li>
        <li>8/9: Steve Daniels</li>
        <li>8/14: Curt Davidson</li>
        <li>8/15: Kay Davis</li>
        <li>8/16: John Davis </li>
        ''',
        missionary1='''
        <a href="https://www.facebook.com/cameron.clayton.3344">Elder Clayton</a>
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
