/**
 * Created by travis.howe on 7/2/18.
 */
function toggleImage() {
  var image = document.getElementById('image');
  image.classList.toggle('hidden');
}

var button = document.getElementById('button')
button.addEventListener('click', toggleImage)
