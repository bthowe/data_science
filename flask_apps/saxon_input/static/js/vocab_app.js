/**
 * Created by travis.howe on 7/2/18.
 */
var cards = {{cards | tojson}}
function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
    }
    return array;
}
cards = shuffle(cards)

var index = 0
document.getElementById("image").src = cards[index][0];

document.addEventListener('DOMContentLoaded', init, false);
function init(){
    function message () {
        document.getElementById("image_hidden").src = cards[index][1];
        var image = document.getElementById('image_hidden');
//            image.classList.toggle('hidden');
        image.className = "unhidden"
    }
    var button_flip = document.getElementById('button_flip');
    button_flip.addEventListener('click', message, true);

    function next () {
        index++;
        var image = document.getElementById('image_hidden');
        if (image.className == "unhidden") {
            image.className = "hidden"
        }
        document.getElementById("image").src = cards[index][0];

        var button_back = document.getElementById('button_back');
        if (index == 1) {
            button_back.className = "unhidden"
        }
        if (index == (cards.length - 1)) {
            button_next.className = "hidden"
        }
    }
    var button_next = document.getElementById('button_next');
    button_next.addEventListener('click', next, true);

    function back () {
        index--;
        var image = document.getElementById('image_hidden');
        if (image.className == "unhidden") {
            image.className = "hidden"
        }
        document.getElementById("image").src = cards[index][0];

        var button_next = document.getElementById('button_next');
        if (index == (cards.length - 2)) {
            button_next.className = "unhidden"
        }
        if (index == 0) {
            button_back.className = "hidden"
        }
    }
    var button_back = document.getElementById('button_back');
    button_back.addEventListener('click', back, true);
};