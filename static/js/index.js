var counterBack;

function startTimer()
{
    clearInterval(counterBack);
    $('.progress-bar').css('width', '100%');
    var seconds = 90;
    var secondsRemaining = seconds
    counterBack = setInterval(function(){
    secondsRemaining--;
    if (secondsRemaining >= 0){
        $('.progress-bar').css('width', (100*secondsRemaining/seconds)+'%');
    } else {
        clearInterval(counterBack);
        document.getElementById('alert-audio').play();
    }

    }, 1000);
}

function resetTimer()
{
    clearInterval(counterBack);
    $('.progress-bar').css('width', '100%');
    document.getElementById('alert-audio').pause();
}