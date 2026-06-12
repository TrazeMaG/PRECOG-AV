"""
PRECOG — Final publishing-grade demo video
Nikhil Upadhyay | Independent Researcher | Dublin Business School
All numbers spelled as words for natural narration. Per-section audio sync.
"""
import cv2, numpy as np, sys, torch, imageio, os, tempfile, asyncio, shutil
from ultralytics import YOLO
sys.path.insert(0, "D:/precog")
from precog_pipeline import PRECOG, DEVICE

CLIP          = "D:/precog/demo_clips/danger/3b938e26-4310-42b7-a0fe-5573e6b9214d.mp4"
OUTPUT_SILENT = "D:/precog/precog_demo_silent.mp4"
OUTPUT_FINAL  = "D:/precog/precog_demo_60s.mp4"
DANGER_FRAME  = 175
DISP_WARN     = 0.45
DISP_CRIT     = 0.68
OUT_FPS       = 30
W, H          = 1920, 1080
FONT          = cv2.FONT_HERSHEY_SIMPLEX

SEC = {"title":3,"problem":4,"clean":3,"alert":8,"freeze":5,"averted":3,"stats":7}

ROAD_CLASSES = {0:"person",1:"bicycle",2:"car",3:"motorcycle",5:"bus",7:"truck"}
CLASS_DANGER = {"person":1.0,"bicycle":0.85,"motorcycle":0.8,
                "car":0.5,"truck":0.45,"bus":0.45}

def n(k): return int(SEC[k]*OUT_FPS)
def tw(t,sc,tk=1): (w,_),_=cv2.getTextSize(t,FONT,sc,tk); return w
def put(f,t,x,y,sc=1.0,col=(255,255,255),tk=2,sh=True):
    if sh: cv2.putText(f,t,(x+2,y+2),FONT,sc,(0,0,0),tk+2)
    cv2.putText(f,t,(x,y),FONT,sc,col,tk)
def put_c(f,t,y,sc=1.0,col=(255,255,255),tk=2):
    put(f,t,max(20,(W-tw(t,sc,tk))//2),y,sc,col,tk)
def bar(f,x,y,w,h,val,col,bg=(40,40,40)):
    cv2.rectangle(f,(x,y),(x+w,y+h),bg,-1)
    cv2.rectangle(f,(x,y),(x+int(np.clip(val,0,1)*w),y+h),col,-1)
def dc(dp): return (0,0,220) if dp>=DISP_CRIT else (0,140,255) if dp>=DISP_WARN else (0,200,80)
def dl(dp): return "CRITICAL" if dp>=DISP_CRIT else "WARNING" if dp>=DISP_WARN else "NOMINAL"
def get_frame(raw,t):
    t=max(0.0,min(float(t),len(raw)-1.001))
    i0=int(t);i1=min(i0+1,len(raw)-1);a=t-i0
    if a<0.01: return raw[i0].copy()
    return cv2.addWeighted(raw[i0],1.0-a,raw[i1],a,0)

def draw_risk_panel(f,dp,ss,n_obj,in_path,prox):
    PX,PY,PW,PH=W-295,20,275,255
    ov=f.copy()
    cv2.rectangle(ov,(PX-2,PY-2),(PX+PW+2,PY+PH+2),(0,0,0),-1)
    cv2.addWeighted(ov,0.75,f,0.25,0,f)
    bc=dc(dp); cv2.rectangle(f,(PX,PY),(PX+PW,PY+PH),bc,2)
    W2=(220,220,220); RED=(0,0,220); ORA=(0,140,255); GRN=(0,200,80)
    put(f,"RISK ANALYSIS",PX+10,PY+22,0.58,W2,1)
    cv2.line(f,(PX+6,PY+30),(PX+PW-6,PY+30),(70,70,70),1)
    y=PY+48
    for label,val,col,fmt in [
        ("Threat prox.",  prox,          RED if prox>0.65 else ORA if prox>0.4 else GRN, lambda v:f"{int(v*100)}%"),
        ("Path status",   float(in_path),RED if in_path else GRN,                         lambda v:"BLOCKED" if v>0.5 else "CLEAR"),
        ("Sensor health", ss,            GRN if ss>0.5 else RED,                          lambda v:f"{int(v*100)}%"),
    ]:
        put(f,label,PX+10,y,0.5,W2,1)
        bar(f,PX+10,y+7,PW-65,13,val,col)
        txt=fmt(val)
        tc=RED if (label=="Path status" and val>0.5) else GRN if label=="Path status" else (GRN if val>0.6 else ORA if val>0.3 else RED)
        put(f,txt,PX+PW-60,y,0.46,tc,1); y+=48
    cv2.line(f,(PX+6,y),(PX+PW-6,y),(70,70,70),1); y+=14
    put(f,"HERALD score",PX+10,y,0.52,W2,1)
    bar(f,PX+10,y+9,PW-20,15,dp,bc)
    put(f,f"{dp:.3f}",PX+PW-60,y,0.5,bc,1); y+=44
    put(f,"SENSE score",PX+10,y,0.52,W2,1)
    sc=GRN if ss>0.5 else RED
    bar(f,PX+10,y+9,PW-20,15,ss,sc)
    put(f,f"{ss:.3f}",PX+PW-60,y,0.5,sc,1)
    return f

def draw_hud(frame,dp,ss,highlight=None,fidx=0,n_obj=0,in_path=False,
             prox=0.3,show_activated=False,ctx=""):
    f=frame.copy(); col=dc(dp); lbl=dl(dp)
    sc=(0,200,80) if ss>0.5 else (0,0,220)
    if highlight and dp>DISP_WARN:
        x1,y1,x2,y2,rl=highlight
        pulse=2+int(4*abs(np.sin(fidx*0.3)))
        glow=f.copy()
        cv2.rectangle(glow,(x1-10,y1-10),(x2+10,y2+10),(0,0,140),pulse+8)
        cv2.addWeighted(glow,0.35,f,0.65,0,f)
        cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,255),pulse)
        lw=tw(rl,0.58,2)
        cv2.rectangle(f,(x1,y1-36),(x1+lw+14,y1),(0,0,180),-1)
        cv2.rectangle(f,(x1,y1-36),(x1+lw+14,y1),(0,0,255),1)
        put(f,rl,x1+7,y1-9,0.58,(255,255,255),2,sh=False)
        cx_o=(x1+x2)//2
        if y1>72: cv2.arrowedLine(f,(cx_o,y1-65),(cx_o,y1-42),(0,0,255),3,tipLength=0.4)
    cv2.rectangle(f,(0,0),(W-1,H-1),col,12 if dp>DISP_WARN else 3)
    ov=f.copy()
    cv2.rectangle(ov,(20,20),(420,240),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,f,0.35,0,f)
    cv2.rectangle(f,(20,20),(420,240),col,2)
    put(f,f"PRECOG  {lbl}",32,68,1.1,col,2)
    put(f,f"DANGER  {int(dp*100)}%",32,104,0.65,(200,200,200),1)
    bar(f,32,113,374,16,dp,col)
    put(f,f"SENSE   {ss:.3f}",32,148,0.65,(200,200,200),1)
    bar(f,32,157,374,16,ss,sc)
    put(f,"RADAR: NOMINAL" if ss>0.5 else "RADAR: DEGRADED",32,193,0.55,sc,1)
    put(f,f"{n_obj} objects   Path: {'BLOCKED' if in_path else 'CLEAR'}",32,218,0.48,(180,180,180),1)
    badge="ViT-B/16  >>  SENSE  >>  HERALD"
    bw=tw(badge,0.5,1)+20
    cv2.rectangle(f,(20,H-65),(20+bw,H-42),(0,0,0),-1)
    cv2.rectangle(f,(20,H-65),(20+bw,H-42),(80,80,80),1)
    put(f,badge,30,H-48,0.5,(140,140,140),1)
    if ctx:
        ctw=tw(ctx,0.62,1)+20
        cv2.rectangle(f,(20,H-38),(20+ctw,H-15),(0,0,0),-1)
        put(f,ctx,30,H-20,0.62,(0,210,80),1)
    if show_activated:
        bw2=480; bh=46; bx=(W-bw2)//2; by=H-78
        cv2.rectangle(f,(bx,by),(bx+bw2,by+bh),(0,140,0),-1)
        cv2.rectangle(f,(bx,by),(bx+bw2,by+bh),(0,255,80),2)
        put_c(f,"PRECOG  ACTIVATED",by+32,0.9,(255,255,255),2)
    draw_risk_panel(f,dp,ss,n_obj,in_path,prox)
    return f

def analyze(res):
    if not res or res[0].boxes is None: return None,0,False,0.1
    boxes=res[0].boxes; best,bs,n_o,ip=None,-1,0,False; dists=[]
    for i in range(len(boxes)):
        cid=int(boxes.cls[i].item()); conf=float(boxes.conf[i].item())
        if cid not in ROAD_CLASSES or conf<0.20: continue
        cls=ROAD_CLASSES[cid]
        x1,y1,x2,y2=[int(v) for v in boxes.xyxy[i].cpu().numpy()]
        cx_n=((x1+x2)/2)/W; cy_n=((y1+y2)/2)/H
        bw_n=(x2-x1)/W; area=((x2-x1)*(y2-y1))/(W*H)
        if cy_n>0.82 and bw_n>0.45: continue
        if y2>H*0.94: continue
        n_o+=1
        if 0.15<cx_n<0.85 and cy_n>0.25: ip=True
        dists.append(max(0.05,1.0-min(area*5,0.95)))
        s=CLASS_DANGER.get(cls,0.4)*0.45+(1-abs(cx_n-0.5))*0.3+min(area*8,1)*0.25
        if s>bs:
            bs=s; rl="! PEDESTRIAN" if cls=="person" else f"! {cls.upper()}"
            best=(x1,y1,x2,y2,rl)
    prox=1.0-min(dists) if dists else 0.1
    return best,n_o,ip,prox

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading PRECOG..."); precog=PRECOG()
print("Loading YOLOv8n..."); yolo=YOLO("yolov8n.pt")
print("Loading clip...")
cap=cv2.VideoCapture(CLIP); fps_src=cap.get(cv2.CAP_PROP_FPS)
raw=[]
while True:
    ret,fr=cap.read()
    if not ret: break
    raw.append(cv2.resize(fr,(W,H)))
cap.release()
print(f"Loaded {len(raw)} frames")

print("Extracting ambient audio...")
ambient_path=None
try:
    try:
        from moviepy import VideoFileClip as VFC
    except ImportError:
        from moviepy.editor import VideoFileClip as VFC
    orig=VFC(CLIP)
    if orig.audio is not None:
        ambient_path="D:/precog/clip_ambient.mp3"
        orig.audio.write_audiofile(ambient_path,logger=None)
    orig.close()
except Exception:
    pass

if not ambient_path or not os.path.exists(str(ambient_path)):
    import scipy.io.wavfile as wf
    RATE=44100; dur=sum(SEC.values())+2
    t=np.linspace(0,dur,int(dur*RATE))
    amb=0.10*np.sin(2*np.pi*65*t)+0.06*np.sin(2*np.pi*130*t)+0.04*np.sin(2*np.pi*260*t)
    amb+=np.random.randn(len(t))*0.025
    ambient_path="D:/precog/clip_ambient.wav"
    wf.write(ambient_path,RATE,np.clip(amb,-1,1).astype(np.float32))
    print("  Synthetic road ambient generated")

print("Computing danger scores...")
buf,scores=[],[]
for fr in raw:
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    buf.append(precog.preprocess(rgb))
    if len(buf)>5: buf.pop(0)
    if len(buf)<5: scores.append(0.0); continue
    with torch.no_grad():
        v=precog.vit(torch.stack(buf).to(DEVICE)).cpu()
        dp=float(torch.sigmoid(precog.herald(
            v.unsqueeze(0).to(DEVICE),torch.zeros(1,7).to(DEVICE))).item())
    scores.append(dp)

SS=0.019
ft=next((i for i,s in enumerate(scores) if s>0.55),DANGER_FRAME)
gap=(DANGER_FRAME-ft)/fps_src
print(f"Trigger: frame {ft}, gap={gap:.1f}s")

ALERT_START=100; SLOW=0.7
alert_t=[ALERT_START+i*SLOW for i in range(n("alert"))]

print("YOLO detections...")
yolo_ids=list(set([int(t) for t in alert_t]+list(range(DANGER_FRAME,min(DANGER_FRAME+6,len(raw))))))
det={}
for idx in yolo_ids:
    rgb=cv2.cvtColor(raw[idx],cv2.COLOR_BGR2RGB)
    det[idx]=analyze(yolo(rgb,verbose=False,classes=list(ROAD_CLASSES.keys())))

out_frames=[]

# 1 TITLE
for _ in range(n("title")):
    f=np.zeros((H,W,3),np.uint8); cv2.rectangle(f,(0,0),(W,H),(10,10,20),-1)
    put_c(f,"PRECOG",H//2-55,3.2,(255,255,255),5)
    put_c(f,"Detects road hazards. Calculates risk. Averts danger.",H//2+20,0.92,(180,180,180),2)
    put_c(f,"Proactive safety AI for autonomous vehicles",H//2+65,0.82,(120,120,120),1)
    put_c(f,"Nikhil Upadhyay  |  Independent Researcher  |  Dublin Business School",H//2+120,0.78,(160,160,160),2)
    put_c(f,"298,326 clips  |  25 countries  |  github.com/TrazeMaG/PRECOG-AV",H//2+165,0.65,(100,100,100),1)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# 2 PROBLEM
for i in range(n("problem")):
    f=np.zeros((H,W,3),np.uint8); cv2.rectangle(f,(0,0),(W,H),(14,10,10),-1)
    put_c(f,"The problem with AV safety today:",H//2-130,0.9,(180,180,180),2)
    put_c(f,"Systems detect danger only after it becomes visible.",H//2-70,0.95,(200,80,80),2)
    put_c(f,"By then, reaction time is critical.",H//2-15,0.9,(200,80,80),2)
    cv2.line(f,(W//4,H//2+45),(3*W//4,H//2+45),(60,60,60),1)
    put_c(f,"PRECOG anticipates danger before it is geometrically measurable.",H//2+88,0.9,(50,200,80),2)
    put_c(f,"From camera pixels alone. No radar. No LiDAR.",H//2+138,0.82,(50,200,80),1)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# 3 CLEAN
for i in range(n("clean")):
    src=min(i,len(raw)-1); f=raw[src].copy()
    put(f,"Standard dashcam  |  No alerts",40,H-50,0.85,(200,200,200),2)
    put(f,"PRECOG: STANDBY",W-360,55,0.85,(100,100,100),2)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# 4 ALERT
activated_shown=False
for i in range(n("alert")):
    t_frac=alert_t[i]; src_idx=int(t_frac)
    dp=scores[src_idx] if src_idx<len(scores) else 0.5
    hl,n_o,ip,prox=det.get(src_idx,(None,0,False,0.1))
    show_act=(src_idx>=ft and not activated_shown)
    if src_idx>=ft: activated_shown=True
    ctx=""
    if src_idx>=DANGER_FRAME: ctx="Pedestrian entering path"
    elif src_idx>=ft: ctx="Danger score rising"
    f=draw_hud(get_frame(raw,t_frac),dp,SS,highlight=hl,fidx=i,
               n_obj=n_o,in_path=ip,prox=prox,show_activated=show_act,ctx=ctx)
    put(f,f"T={t_frac/fps_src:.1f}s",W-580,H-48,0.8,(180,180,180),1)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# 5 FREEZE
fsrc=min(DANGER_FRAME,len(raw)-1)
fdp=scores[fsrc] if fsrc<len(scores) else 0.72
fhl,fn_o,fip,fprox=det.get(fsrc,(None,0,False,0.5))
base=draw_hud(raw[fsrc],fdp,SS,highlight=fhl,fidx=0,n_obj=fn_o,in_path=fip,prox=fprox)
for i in range(n("freeze")):
    f=base.copy(); alp=min(i/(OUT_FPS*0.5),1.0)
    if alp>0.2:
        ov=f.copy()
        cv2.rectangle(ov,(W//2-570,H//2-150),(W//2+570,H//2+230),(0,0,0),-1)
        cv2.addWeighted(ov,0.82*alp,f,1-0.82*alp,0,f)
        put_c(f,f"Danger condition reached:  T = {DANGER_FRAME/fps_src:.1f}s",H//2-90,1.0,(60,60,220),2)
        put_c(f,f"PRECOG first triggered:   T = {ft/fps_src:.1f}s",H//2-22,1.0,(50,200,80),2)
        cv2.line(f,(W//2-530,H//2+32),(W//2+530,H//2+32),(70,70,70),1)
        put_c(f,f"{gap:.1f}  seconds early warning",H//2+122,2.2,(255,255,255),4)
        put_c(f,"Camera only  |  Radar DEGRADED  |  No LiDAR",H//2+202,0.85,(140,140,140),1)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# 6 AVERTED
avert_src=raw[min(len(raw)-1,DANGER_FRAME+35)].copy()
for i in range(n("averted")):
    f=avert_src.copy(); alp=min(i/(OUT_FPS*0.35),1.0)
    ov=f.copy(); cv2.rectangle(ov,(0,0),(W,H),(0,45,0),-1)
    cv2.addWeighted(ov,0.42*alp,f,1-0.42*alp,0,f)
    cv2.rectangle(f,(0,0),(W-1,H-1),(0,200,80),8)
    put_c(f,"DANGER  AVERTED",H//2-28,3.0,(0,220,80),5)
    put_c(f,"PRECOG identified the threat before it reached critical range",H//2+68,0.85,(180,180,180),2)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# 7 STATS
rows=[
    ("Method",        "CCD AP",   "DAD mTTA","Countries",(200,200,200)),
    ("DSA  (2016)",   "-",        "1.67s",   "1",        (130,130,130)),
    ("GCRN (2020)",   "-",        "2.33s",   "1",        (130,130,130)),
    ("DSTA (2021)",   "-",        "2.55s",   "1",        (130,130,130)),
    ("LATTE (2025)",  "-",        "3.16s",   "1",        (130,130,130)),
    ("RARE (2025)",   "99.80%",   "-",       "1",        (130,130,130)),
    ("PRECOG (ours)", "99.95% *", "3.83s *", "25",       (50,200,80)),
]
COL_X=[80,540,880,1260]
for _ in range(n("stats")):
    f=np.zeros((H,W,3),np.uint8); cv2.rectangle(f,(0,0),(W,H),(10,10,18),-1)
    put_c(f,"PRECOG vs State of the Art",110,1.6,(255,255,255),3)
    y=215
    for row in rows:
        col=row[4]; is_us="ours" in row[0]; is_hd=row[0]=="Method"
        s=0.88 if is_us else 0.76; tk=2 if is_us else 1
        if is_us: cv2.rectangle(f,(60,y-40),(W-60,y+18),(0,40,0),-1)
        if is_hd: cv2.line(f,(60,y+20),(W-60,y+20),(60,60,60),1)
        for xi,txt in zip(COL_X,row[:4]): put(f,str(txt),xi,y,s,col,tk)
        y+=62
    put_c(f,"* = best published result on benchmark",H-90,0.75,(90,90,90),1)
    put_c(f,"github.com/TrazeMaG/PRECOG-AV  |  NVIDIA PhysicalAI-AV  |  Nikhil Upadhyay",H-50,0.72,(70,70,70),1)
    out_frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))

# ── Write silent video ─────────────────────────────────────────────────────────
total_s=len(out_frames)/OUT_FPS
print(f"\nWriting {len(out_frames)} frames ({total_s:.1f}s)...")
writer=imageio.get_writer(OUTPUT_SILENT,fps=OUT_FPS,codec="libx264",quality=8,
    macro_block_size=1,ffmpeg_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
for frm in out_frames: writer.append_data(frm)
writer.close()
print(f"Silent: {os.path.getsize(OUTPUT_SILENT)/1e6:.1f}MB")

# ── Per-section synced narration — ALL numbers as words ───────────────────────
print("\nGenerating narration (numbers as words, per-section sync)...")
VOICE="en-US-GuyNeural"

print("\nGenerating narration (per-section synced)...")
VOICE="en-US-GuyNeural"

# Plain text only — no SSML, no numbers spoken
section_scripts={
    "title":   "PRECOG. Detects road hazards. Calculates risk. Averts danger.",
    "problem": "Today's autonomous vehicles only react after danger becomes visible. PRECOG sees it coming first.",
    "clean":   "Standard dashcam. Normal city driving. No alerts.",
    "alert_a": "Detecting threat.",        # fires at T=0 of alert section
    "alert_b": "Calculating safe path.",   # fires at T=3.8s of alert section
    "freeze":  "Early warning delivered.",
    "averted": "Danger averted.",
    "stats":   "Best published results. Car Crash Dataset. D-A-D benchmark. Twenty-five countries.",
}
# Section order for concatenation — alert split into two sub-clips
SECTION_ORDER=["title","problem","clean","alert","freeze","averted","stats"]

async def tts(text, voice, rate, path):
    import edge_tts
    c=edge_tts.Communicate(text, voice, rate=rate)
    await c.save(path)

try:
    try:
        from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
        from moviepy.audio.AudioClip import AudioArrayClip, concatenate_audioclips
    except ImportError:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        from moviepy.audio.AudioClip import AudioArrayClip, concatenate_audioclips

    RATE=44100
    def silence(dur):
        n=max(1,int(dur*RATE))
        return AudioArrayClip(np.zeros((n,2),dtype=np.float32),fps=RATE)

    def make_clip(key, text, target_dur, rate="+20%"):
        pf=tempfile.NamedTemporaryFile(suffix=".mp3",delete=False); pf.close()
        asyncio.run(tts(text,VOICE,rate,pf.name))
        a=AudioFileClip(pf.name)
        dur=a.duration
        try: os.remove(pf.name)
        except: pass
        if dur<target_dur:
            return concatenate_audioclips([a, silence(target_dur-dur)])
        return a.subclip(0,target_dur)

    clips=[]
    for sec_key in SECTION_ORDER:
        target=SEC[sec_key]

        if sec_key=="alert":
            # Two phrases with programmatic silence between
            # "Detecting threat." at start, silence, "Calculating safe path." at 3.8s
            clip_a_tmp=tempfile.NamedTemporaryFile(suffix=".mp3",delete=False); clip_a_tmp.close()
            asyncio.run(tts(section_scripts["alert_a"],VOICE,"+20%",clip_a_tmp.name))
            ca=AudioFileClip(clip_a_tmp.name)

            clip_b_tmp=tempfile.NamedTemporaryFile(suffix=".mp3",delete=False); clip_b_tmp.close()
            asyncio.run(tts(section_scripts["alert_b"],VOICE,"+20%",clip_b_tmp.name))
            cb=AudioFileClip(clip_b_tmp.name)

            # Layout: [phrase_a][gap to 3.8s][phrase_b][remaining silence to 8s]
            gap_dur=max(0.1, 3.8-ca.duration)
            after_dur=max(0.1, target-3.8-cb.duration)
            alert_clip=concatenate_audioclips([
                ca,
                silence(gap_dur),
                cb,
                silence(after_dur),
            ])
            print(f"  alert: '{section_scripts['alert_a']}' + {gap_dur:.1f}s gap + '{section_scripts['alert_b']}' + {after_dur:.1f}s → {alert_clip.duration:.1f}s")
            clips.append(alert_clip.subclip(0,target) if alert_clip.duration>target else alert_clip)
            for p in [clip_a_tmp.name, clip_b_tmp.name]:
                try: os.remove(p)
                except: pass
        else:
            text=section_scripts.get(sec_key,"")
            if not text:
                clips.append(silence(target))
                continue
            # Measure at +15%
            pf=tempfile.NamedTemporaryFile(suffix=".mp3",delete=False); pf.close()
            asyncio.run(tts(text,VOICE,"+15%",pf.name))
            a=AudioFileClip(pf.name); dur_15=a.duration; a.close()
            try: os.remove(pf.name)
            except: pass
            natural=dur_15*1.15
            rate=f"+{max(15,int((natural/target-1)*100))}%" if natural>target else "+15%"
            clips.append(make_clip(sec_key,text,target,rate))
            print(f"  {sec_key}: {natural:.1f}s natural → {rate} → target {target}s")

    narr=concatenate_audioclips(clips)
    print(f"  Total: {narr.duration:.1f}s == video {total_s:.1f}s")
    narr_tmp=tempfile.NamedTemporaryFile(suffix=".mp3",delete=False); narr_tmp.close()
    narr.write_audiofile(narr_tmp.name,logger=None)
    narr.close()

    vid=VideoFileClip(OUTPUT_SILENT)
    na=AudioFileClip(narr_tmp.name)

    if ambient_path and os.path.exists(str(ambient_path)):
        amb=AudioFileClip(str(ambient_path)).volumex(0.18)
        if amb.duration<total_s:
            loops=int(total_s/amb.duration)+2
            try: amb=concatenate_audioclips([amb]*loops).subclip(0,total_s)
            except: amb=amb.subclip(0,min(amb.duration,total_s))
        else:
            amb=amb.subclip(0,total_s)
        mixed=CompositeAudioClip([amb,na.volumex(1.0)])
    else:
        mixed=na

    vid.set_audio(mixed).write_videofile(OUTPUT_FINAL,
        codec="libx264",audio_codec="aac",logger=None)
    vid.close()
    try: os.remove(narr_tmp.name)
    except: pass
    try: os.remove(OUTPUT_SILENT)
    except: pass
    print(f"Final: {os.path.getsize(OUTPUT_FINAL)/1e6:.1f}MB")

except Exception as e:
    import traceback
    print(f"Audio failed: {e}"); traceback.print_exc()
    shutil.copy(OUTPUT_SILENT,OUTPUT_FINAL)
    print("Silent saved as final.")

print(f"\nDone: {OUTPUT_FINAL}")
print(f"Publishing-grade demo | Nikhil Upadhyay")