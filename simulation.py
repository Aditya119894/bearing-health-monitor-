"""Predictive Bearing Health Monitor - simulation.
Simulated motor vibration with a growing bearing fault. An Isolation Forest
learns healthy vibration, detects the fault, and the motor slows to 40%.
Outputs: motor_pro_fixed.mp4 (dashboard video) and results.png (graph)."""
import numpy as np, imageio, subprocess
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import kurtosis
from PIL import Image, ImageDraw

rng = np.random.default_rng(0)
FS, WIN, BPFO = 2000, 200, 87
W, H = 960, 540

def window(sev):
    t = np.arange(WIN)/FS
    x = rng.normal(0,.05,WIN) + .03*np.sin(2*np.pi*50*t + rng.uniform(0,6.28))
    if sev > 0:
        for t0 in np.arange(rng.uniform(0,1/BPFO), WIN/FS, 1/BPFO):
            m = t >= t0
            x[m] += sev*1.5*np.exp(-(t[m]-t0)*400)*np.sin(2*np.pi*800*(t[m]-t0))
    return x

def feats(x):
    r = np.sqrt(np.mean(x**2)); pk = np.max(np.abs(x))
    return [r, kurtosis(x), pk, pk/r]

healthy = np.array([feats(window(0)) for _ in range(800)])
model = IsolationForest(contamination=0.005, random_state=0).fit(healthy)

ONSET, sev, speed, consec, alarm_t, ang = 12.0, 0.0, 1.0, 0, None, 0.0
wave, scores, frames = [], [], []
log = {"t": [], "rms": [], "score": [], "speed": []}
GREEN, RED, AMBER = (60,220,100), (240,70,70), (250,180,40)

for k in range(400):
    t = k*0.1
    if t >= ONSET: sev = min(1, sev + 0.1*speed/28)
    w = window(sev); f = feats(w)
    score = -model.score_samples([f])[0]
    consec = consec+1 if model.predict([f])[0] == -1 else 0
    if consec >= 4 and alarm_t is None: alarm_t, speed = t, 0.4
    for key, v in zip(log, [t, f[0], score, speed]): log[key].append(v)
    wave.extend(w[::4]); wave = wave[-300:]
    scores.append(score); scores = scores[-120:]
    ang += speed*0.35
    col = GREEN if alarm_t is None else RED

    im = Image.new("RGB", (W,H), (14,16,24)); d = ImageDraw.Draw(im)
    d.rectangle([0,0,W,56], fill=(24,28,42))
    d.text((20,12), "BEARING HEALTH MONITOR  |  ESP32 + MPU6050 + Isolation Forest", fill=(255,255,255))
    d.text((20,32), f"Time {t:5.1f}s    Bearing wear {sev*100:3.0f}%    Motor speed {speed*100:3.0f}%", fill=(180,190,210))

    d.rounded_rectangle([40,170,300,330], 14, fill=(50,56,76), outline=(110,120,150), width=3)
    d.text((60,180), "MOTOR", fill=(200,210,230))
    d.rectangle([300,242,400,258], fill=(150,155,170))
    jx, jy = 340 + rng.normal(0,sev*3), 250 + rng.normal(0,sev*3)
    d.ellipse([jx-38,jy-38,jx+38,jy+38], outline=col, width=5)
    d.text((jx-28,jy+46), "BEARING", fill=col)
    for a in (0,120,240):
        r = ang + np.radians(a)
        d.line([jx,jy,jx+34*np.cos(r),jy+34*np.sin(r)], fill=col, width=4)
    d.ellipse([jx-5,jy-5,jx+5,jy+5], fill=col)
    d.rectangle([400,242,470,258], fill=(150,155,170))
    d.rectangle([470,200,490,300], fill=(110,120,150))

    d.rounded_rectangle([520,80,930,250], 10, fill=(20,24,36), outline=(70,80,110), width=2)
    d.text((532,86), "LIVE VIBRATION (MPU6050)", fill=(180,190,210))
    pts = [(530+i*400/300, 170 - wave[i]*70) for i in range(len(wave))]
    if len(pts) > 1: d.line(pts, fill=col, width=2)

    d.rounded_rectangle([520,270,930,440], 10, fill=(20,24,36), outline=(70,80,110), width=2)
    d.text((532,276), "AI ANOMALY SCORE", fill=(180,190,210))
    thr = 410 - (0.55-0.35)/(0.8-0.35)*130
    d.line([525,thr,925,thr], fill=AMBER, width=1)
    sp = [(530+i*400/120, 425 - np.clip((s-0.35)/0.45,0,1)*130) for i,s in enumerate(scores)]
    if len(sp) > 1: d.line(sp, fill=col, width=2)

    ban = (20,70,40) if alarm_t is None else (110,25,25)
    d.rounded_rectangle([40,470,930,520], 12, fill=ban)
    msg = "STATUS: HEALTHY - monitoring" if alarm_t is None else f"FAULT DETECTED at {alarm_t:.1f}s - MOTOR SLOWED TO 40% - MAINTENANCE ALERT SENT"
    d.text((60,488), msg, fill=(255,255,255))
    frames.append(np.array(im))

imageio.mimsave("motor_pro.mp4", frames, fps=10, macro_block_size=1)
import imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-i","motor_pro.mp4","-c:v","libx264",
                "-pix_fmt","yuv420p","-movflags","+faststart","motor_pro_fixed.mp4"])

# Results graph
fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
ax[0].plot(log["t"], log["rms"]); ax[0].set_ylabel("Vibration RMS")
ax[1].plot(log["t"], log["score"], c="r"); ax[1].set_ylabel("AI anomaly score")
ax[2].plot(log["t"], log["speed"], c="g"); ax[2].set_ylabel("Motor speed")
ax[2].set_xlabel("Time (s)")
for a in ax:
    a.axvline(ONSET, ls="--", c="gray")
    if alarm_t: a.axvline(alarm_t, ls="--", c="k")
plt.tight_layout(); plt.savefig("results.png", dpi=150)

print("Fault began:", ONSET, "s | AI alarm:", alarm_t, "s")
print("Detection delay:", round(alarm_t-ONSET, 1) if alarm_t else None, "s")
print("False alarm before fault:", alarm_t is not None and alarm_t < ONSET)