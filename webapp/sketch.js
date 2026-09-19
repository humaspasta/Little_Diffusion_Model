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


async function loadModel(pattern) {
    const session = await ort.InferenceSession.create(`onnx_files/Onnx_files/${pattern}_model.onnx`)
    return session
}

// runs the model on every sample at once instead of one point at a time
async function runModelBatch(session, x_t, t, T) {
    const n_samples = x_t.length
    const input = new Float32Array(n_samples * 3)

    x_t.forEach(([x, y], i) => {
        input[i * 3] = x
        input[i * 3 + 1] = y
        input[i * 3 + 2] = t / T
    })

    const tensor = new ort.Tensor('float32', input, [n_samples, 3])
    const results = await session.run({ input: tensor })
    const output = results.output.data

    return Array.from({length: n_samples}, (_, i) => [output[i * 2], output[i * 2 + 1]])
}


function sample_two_gaussian() {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)  // gaussian sample 1
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2) //gaussian sample 2
    return [z0 , z1]
}

async function recreate_pattern_dataset(session, T, n_samples=200) {
    const { betas, alphas, alpha_bars } = create_noise_schedule(T)

    // initialize x_t with random gaussian noise — shape (n_samples, 2)
    let x_t = Array.from({length: n_samples}, () => sample_two_gaussian())

    for (let t = T - 1; t >= 0; t--) {
        const z = t >= 1
            ? Array.from({length: n_samples}, () => sample_two_gaussian())
            : Array.from({length: n_samples}, () => [0, 0])

        const model_vals = await runModelBatch(session, x_t, t, T)

        const scale = 1 / Math.sqrt(alphas[t])
        const noise_scale = (1 - alphas[t]) / Math.sqrt(1 - alpha_bars[t])
        const z_scale = Math.sqrt(betas[t])

        x_t = x_t.map(([x, y], i) => {
            return [
                scale * (x - noise_scale * model_vals[i][0]) + z_scale * z[i][0],
                scale * (y - noise_scale * model_vals[i][1]) + z_scale * z[i][1]
            ]
        })

        if (t % 20 === 0) {
            drawPoints(x_t)  // draw intermediate steps so you can watch it denoise
            await sleep(1)  // small delay so browser can render
        }
    }
    return x_t
}


function create_noise_schedule(T) {
    const alpha_bars = []
    const alphas = []
    const betas = []
    const lower = 1e-4
    const upper = 0.02
    const step = (upper - lower) / T
    let accumulator = 1.0

    for(let i = 0; i < T; i++) {
        const beta = lower + step * i
        const alpha = 1 - beta
        accumulator *= alpha
        betas.push(beta)
        alphas.push(alpha)
        alpha_bars.push(accumulator)
    }

    return { betas, alphas, alpha_bars }
}

// helper to pause execution so canvas can update
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

const sessions = {}
const generateBtn = document.getElementById('generate')
const patternSelect = document.getElementById('pattern')

async function getSession(pattern) {
    if (!sessions[pattern]) {
        sessions[pattern] = await loadModel(pattern)
    }
    return sessions[pattern]
}

generateBtn.addEventListener('click', async () => {
    const pattern = patternSelect.value
    generateBtn.disabled = true
    generateBtn.textContent = 'Generating...'
    try {
        const session = await getSession(pattern)
        await recreate_pattern_dataset(session, 1000, 200)
    } catch(err) {
        console.error('error:', err)
    } finally {
        generateBtn.disabled = false
        generateBtn.textContent = 'Generate'
    }
})
