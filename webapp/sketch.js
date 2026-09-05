var c = document.getElementById('viz')
var ctx = c.getContext('2d')


function drawPoints(points){
    ctx.clearRect(0,0,c.width , c.height)
    ctx.fillStyle = 'orange'

    points.forEach(([x,y]) => {
        const px = (x+1) / 2 * c.width 
        const py = (y+1) /2 * c.height 
        ctx.beginPath()
        ctx.arc(px,py,3,0,Math.PI * 2)
        ctx.fill()
    })
}

function sample_two_guassian() {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)  // gaussian sample 1
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2)
    return [z0 , z1]
}

const sample_frame = (pattern , T , n_samples=20) => { 

}



const points = Array.from({length: 200} , ()=> [
    Math.random() * 2 - 1 ,
    Math.random() * 2 - 1 
 ]
)

drawPoints(points)