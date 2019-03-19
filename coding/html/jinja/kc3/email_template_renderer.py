from jinja2 import Template
with open('email_template.html') as file_:
    template = Template(file_.read())
    t = template.render(
        hero_image="https://media.ldscdn.org/images/media-library/prayer/personal-prayer-581962-wallpaper.jpg",
        missionary_message='''
        <p style="margin: 0; padding-top: .5cm">Greetings Kansas City 3rd Ward!</p>
        <p>Luke 1:70 gives the following description of a prophet's role in God’s plan of salvation: “[The Lord] spake by the mouth of his holy prophets, which have been since the world began.” Jesus Christ calls prophets to communicate his will to us.</p>
        <p>Last week during General Conference our prophet, President Russell M. Nelson, invited us to study the messages shared during the conference. He said they “express the mind and the will of the Lord for His people, today.”</p>
        <p>We invite you to heed this invitation from President Nelson by reviewing the conference talks. We know that we all will be blessed by doing so.</p>
        <p>Thank you for all your love and support.</p>
        ''',
        scripture_button='https://www.lds.org/general-conference/2018/10/becoming-exemplary-latter-day-saints?lang=eng',
        scripture_prompt="Read President Nelson's Address",
        tip_of_week='''
        <h3>Obtain the Word</h3>
        <h5>Develop the habit of consistent daily prayer and scripture study that you may obtain the word.</h5>
        <p>The Lord said, “Seek not to declare my word, but first seek to obtain my word, and then shall your tongue be loosed; then, if you desire, you shall have my Spirit and my word, yea, the power of God unto the convincing of men” (Doctrine and Covenants 11:21).</p>
        <p>As you develop the habit of consistent daily prayer and scripture study you will obtain a greater knowledge and understanding of the word of God, your testimony and knowledge will grow "line upon line, precept upon precept, here a little and there a little" (2 Nephi 28:30). This will create confidence, bring you closer to the Holy Ghost, and give you a greater desire to share what you know to be true.</p>
        <p>Remember, you do not need to have a perfect knowledge of the gospel or the Book of Mormon to share it with others, if you are praying and reading daily, you will know enough. As you feel prompted to share the gospel and act on those promptings, the Holy Ghost will bring to your remembrance and will fill your mouth with those things that you have been studying. Because of this, you can and will feel confident in sharing your feelings and knowledge of the gospel.</p>
        <p>~ Sister Pace</p>
        ''',
        # tip_button='''
        # <td style="border-radius: 3px; background: #909497; text-align: center;" class="button-td">
        #     <a href={0} style="background: #909497; border: 15px solid #909497; font-family: sans-serif; font-size: 13px; line-height: 1.1; text-align: center; text-decoration: none; display: block; border-radius: 3px; font-weight: bold;" class="button-a">
        #         <span style="color:#ffffff;" class="button-link"> {1} </span>
        #     </a>
        # </td>
        # '''.format("https://www.mormonchannel.org/listen/series/conversations-audio/michael-otterson-public-affairs-episode-42", "Watch Interview"),
        announcements='''
        <ul>
        <li>This Sunday during sacrament meeting is our ward's Primary program. Feel free to invite your friends to come and listen to the children of our ward share their testimonies.
        <li>Hadley Bakkedahl's baptism is this Saturday at 3 pm at the church building.
        <li>The chili cookoff/trunk-or-treat social is October 27th at 6 pm.
        </ul>
        ''',
        dinners='''
        <li>10/16: Green</li>
        <li>10/17: Gutierrez</li>
        <li>10/18: Ellibee</li>
        <li>10/19: Davis</li>
        <li>10/20: <i>available</i> </li>
        <li>10/21: Puckett</li>
        <li>10/22: Pace</li>
        ''',
        teamups1='''
        <li>10/16: Randy Puckett</li>
        <li>10/17: Ralph Prowell</li>
        <li>10/18: Steven Poor</li>
        ''',
        teamups2='''
        <li>10/23: Jonathan Semones</li>
        <li>10/24: Ashwin Shashindranath</li>
        <li>10/25: Ed Summer</li>
        ''',
        teamups3='''
        <li>10/30: Russ Willoughby</li>
        <li>11/1: Jason Stephens</li>
        ''',
        teamups4='''
        <li>11/6: John Davis</li>
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

# teamups
# Brother Semone

