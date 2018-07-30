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
        <li>ICYMI: Emily McCulloch was baptized and confirmed this past weekend. Congratulations to her and her family! We are excited to have her as part of our ward! </li>
        <li>Beginning this week, members of the Elders Quorum will have periodic team-up assignments with the full-time missionaries. These typically take place in the evening (and last between one and two hours) but can occur in the morning or afternoon if that is better for your schedule. See below for assignments for the next four weeks.</li>
        <li>Over the next few weeks (before summer vacation ends) the full-time missionaries will be visiting and teaching lessons to the young men and young women of the ward. Please welcome them into your home as they try to strengthen our youth and their friends.</li>
        <li>Finally, Bro. Howe will be out of town for the next three Sundays. Please contact Bro. Josephson (contact information found below) if you have a question or comment related to the ward's missionary efforts.</li>


        </ul>
        ''',
        dinners='''
        <li>7/30: Semones</li>
        <li>7/31: Davis</li>
        <li>8/1: <i> available </i></li>
        <li>8/2: <i> available </i></li>
        <li>8/3: Goin</li>
        <li>8/4: <i> available </i></li>
        <li>8/5: Bakkedahl</li>
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
        <li>8/16: John Davis</li>

        <li>8/21: Bill Ellibee</li>
        <li>8/22: Bill Ellis</li>
        <li>8/23: Craig Fitt</li>
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
