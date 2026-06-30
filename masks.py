#imports
from nrclex import NRCLex
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xml.etree.ElementTree as ET
import colorsys

text_object = NRCLex()

#get logs
directory = Path("logs")

def getInternalResponses(direct):
    responses = []
    for file_path in direct.iterdir():
        if file_path.is_file():
            tree = ET.parse(file_path)
            root = tree.getroot()
            log_el = root.find('.//response')
            if log_el is not None:
                responses.append(log_el.text or "")
            else:
                print(f"No <response> element found in {file_path}")
    return ", ".join(responses)

#more positive or negative?
def getVibe():
    freq = text_object.affect_frequencies
    if freq.get("positive", 0) >= freq.get("negative", 0):
        return "positive"
    else:
        return "negative"

#readable top 3 emotions
def getEmotions():
    result = []
    successCount = 0
    for k, v in freqSort:
        if k != "positive" and k != "negative":
            result.append((k, f"{round(v*100,2)}%"))
            successCount += 1
        if successCount == 3:
            break
    return result

#readable top emotions (including positive and negative)
def topEmotionPercentage():
    list = []
    for k, v in freqSort:
        vPercent = round(v*100, 2)
        tuple = (k, f"{vPercent}%")
        list.append(tuple)
    return list

#create graph
def graph():
    emotions = list(dict(freqSort).keys())
    scores = list(dict(freqSort).values())
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=scores, y=emotions, palette="viridis", hue=emotions, legend=False)
    # 5. Format the chart
    plt.title("Normalized Affect Frequencies", fontsize=16, pad=15, weight='bold')
    plt.xlabel("Proportional Score (0.0 to 1.0)", fontsize=12, labelpad=10)
    plt.ylabel("Emotions / Sentiments", fontsize=12)
    plt.xlim(0, 1.0)  # Frequencies are always bounded between 0 and 1

    for p in ax.patches:
        width = p.get_width()
        if width > 0:  # Only label bars with a value
            ax.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.2f}',
                va='center', ha='left', fontsize=10,)

    plt.tight_layout()
    plt.show()


#face color determined by stereotypical associations combining
#green is disgust, red is anger, blue is sadness
def calculateRGB():
    freq = text_object.affect_frequencies
    greenMult = round(freq.get("disgust", 1)*10, 2)
    redMult = round(freq.get("anger", 1)*10, 2)
    blueMult = round(freq.get("sadness", 1)*10,2)
    r = min(255, 255 * redMult)
    g = min(255, 255 * greenMult)
    b = min(255, 255 * blueMult)

    adjusted = joyMult(r, g, b)

    return adjusted[0], adjusted [1], adjusted[2]


def complementary():
    r = 255 - calculateRGB()[0]
    g = 255 - calculateRGB()[1]
    b = 255 - calculateRGB()[2]

    adjusted = joyMult(r, g, b)

    return adjusted[0], adjusted [1], adjusted[2]

def accentRGB():
    colors = calculateRGB()
    h, l, s = colorsys.rgb_to_hls(colors[0] / 255, colors[1] / 255, colors[2] / 255)
    l = max(0.0, l * 0.90)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r = r * 255
    g = g * 255
    b = b * 255
    result = f'rgb({r},{g},{b})'
    return result


def joyMult(r, g, b):
    freq = text_object.affect_frequencies
    joyBoost = round(freq.get("joy", 0) * 10, 2)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    s = min(1.0, s + joyBoost * 0.1)
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    fr = int(r * 255)
    fg = int(g * 255)
    fb = int(b * 255)

    return fr, fg, fb

#eye width determined by trust and surprise
#narrower eyes = "at ease", heightened eyes = "shocked
def createEyes():
    freq = text_object.affect_frequencies
    xMult = round(freq.get("trust", 1)*5,2)
    yMult = round(freq.get("surprise", 1)*5,2)
    eyeX = xMult * 50
    eyeY = yMult * 50
    return eyeX, eyeY

#outer shape rounder for joy, sharper for anticipativeness
def maskCurvature():
    freq = text_object.raw_emotion_scores
    joyVal = freq.get("joy", 1) * 5
    antVal = freq.get("anticipation", 1) * 5
    result = f'rx="{antVal}" ry="{joyVal}"'
    return result

#difference between most common and least common emotion
def facialSymmetry():
    sortRaw = sorted(text_object.raw_emotion_scores.items(), key=lambda x: x[1], reverse=True)
    diff = (sortRaw[0][1] - sortRaw[-1][1]) * 0.1
    return diff

# changes based on most prominent emotion
# color is slightly lighter version of color selected for mask itself
# pattern.monster <- source
def selectPattern():
    fillColor = f'rgb({calculateRGB()[0]}, {calculateRGB()[1]}, {calculateRGB()[2]})'
    if getEmotions()[0][0] == "fear":
        patternString = f'<pattern id="fear" width="80" height="20" patternTransform="rotate(130)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="none" stroke="{accentRGB()}" stroke-width="4.5" d="M-20.133 4.568C-13.178 4.932-6.452 7.376 0 10s13.036 5.072 20 5c6.967-.072 13.56-2.341 20-5s13.033-4.928 20-5c6.964-.072 13.548 2.376 20 5s13.178 5.068 20.133 5.432"/></pattern>'
        patId = "fear"
    elif getEmotions()[0][0] == "trust":
        patternString = f'<pattern id="trust" width="26" height="20" patternTransform="rotate(135)scale(2)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="none" stroke="{accentRGB()}" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M6.25 10h13.5M13 3.25v13.5"/></pattern>'
        patId = "trust"
    elif getEmotions()[0][0] == "sadness":
        patternString = f'<pattern id="sadness" width="15.825" height="26.667" patternTransform="scale(2)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="{accentRGB()}" d="M-3.176 15.632a1.5 1.5 0 0 0-.294.038 1.463 1.463 0 0 0-1.08 1.754l.013.05c.503 2.134 1.828 3.999 3.533 5.201a9.2 9.2 0 0 0 5.803 1.68c2.012-.098 3.962-.883 5.422-2.17a8.1 8.1 0 0 0 1.93-2.494 9 9 0 0 0 2.67 2.984 9.2 9.2 0 0 0 5.803 1.68c2.012-.098 3.962-.883 5.422-2.17 1.472-1.277 2.454-3.068 2.7-4.944a.22.22 0 0 0-.16-.234c-.11-.036-.221.037-.246.148a7.3 7.3 0 0 1-2.932 4.207 7.6 7.6 0 0 1-4.772 1.325c-1.656-.098-3.227-.76-4.392-1.815-1.178-1.043-1.938-2.478-2.098-3.95a.4.4 0 0 0-.036-.172 1.463 1.463 0 0 0-1.755-1.08 1.463 1.463 0 0 0-1.079 1.754l.012.05q.183.769.5 1.484a7.35 7.35 0 0 1-2.205 2.404 7.6 7.6 0 0 1-4.772 1.325c-1.656-.098-3.227-.76-4.392-1.815-1.178-1.043-1.938-2.478-2.098-3.95a.4.4 0 0 0-.036-.172 1.464 1.464 0 0 0-1.461-1.118M-11.51 2.298a1.463 1.463 0 0 0-1.373 1.792l.013.05c.503 2.135 1.828 4 3.533 5.202a9.2 9.2 0 0 0 5.802 1.68c2.012-.098 3.962-.883 5.422-2.171a8.1 8.1 0 0 0 1.931-2.493 9 9 0 0 0 2.67 2.983 9.2 9.2 0 0 0 5.802 1.68c2.012-.097 3.962-.882 5.422-2.17 1.473-1.276 2.454-3.067 2.7-4.944a.22.22 0 0 0-.16-.233c-.11-.037-.22.037-.245.147a7.3 7.3 0 0 1-2.933 4.208 7.6 7.6 0 0 1-4.771 1.325c-1.656-.098-3.227-.76-4.392-1.816-1.178-1.043-1.939-2.478-2.098-3.95a.4.4 0 0 0-.037-.172 1.463 1.463 0 0 0-1.754-1.08 1.463 1.463 0 0 0-1.08 1.755l.013.05c.12.512.29 1.007.5 1.483A7.35 7.35 0 0 1 1.25 8.03a7.6 7.6 0 0 1-4.773 1.325c-1.656-.098-3.226-.76-4.392-1.816-1.177-1.043-1.938-2.478-2.097-3.95a.4.4 0 0 0-.037-.172 1.464 1.464 0 0 0-1.46-1.118z"/></pattern>'
        patId = "sadness"
    elif getEmotions()[0][0] == "joy":
        patternString = f'<pattern id="joy" width="80" height="30" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="#{fillColor}"/><path fill="none" stroke="{accentRGB()}" stroke-width="1.5" d="M-20.133 4.568C-13.178 4.932-6.452 7.376 0 10s13.036 5.072 20 5c6.967-.072 13.56-2.341 20-5s13.033-4.928 20-5c6.964-.072 13.548 2.376 20 5s13.178 5.068 20.133 5.432"/></pattern>'
        patId = "joy"
    elif getEmotions()[0][0] == "surprise":
        patternString = f'<pattern id="surprise" width="20" height="23" patternTransform="scale(2)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="none" stroke="{accentRGB()}" stroke-linecap="square" d="M-5 5 5.1 15 15 5l10 10"/></pattern>'
        patId = "surprise"
    elif getEmotions()[0][0] == "anger":
        patternString = f'<pattern id="anger" width="58" height="100.23" patternTransform="scale(8)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="none" stroke="{accentRGB()}" stroke-linecap="square" stroke-width=".5" d="m.111-33.307-28.997 16.744zm.012.006 28.993 16.738-.004 33.485L.115 33.492l-28.997-16.57.004-33.485m40.992 43.198v-5.672l4.937 2.85M29.113 9.995 12.117.18l17-9.815M6.114 30.062V10.57l16.967 9.798m-51.963-3.446 57.998-33.485m-29 50.055-.005-66.8m29.001 50.23-57.99-33.485m57.992 19.63-5-2.887 5.002-2.887m28.872-30.805L28.99-16.768zm.012.006 28.993 16.738-.004 33.485-28.997 16.57-28.997-16.57.004-33.485m-.004 33.485 57.998-33.485M57.992 33.287l-.004-66.799m29 50.229Q57.928-.065 28.999-16.768M28.998 2.86l4.998-2.886-4.998-2.886m6.029 23.076 16.964-9.794.002 19.49m-6-3.43v-5.67l-4.936 2.85M28.995 9.789 45.994-.026 28.998-9.84M-.003 66.943-29 83.687zm.012.006 28.993 16.738-.004 33.485m0 0L.001 133.742m0 0-28.997-16.57m0 0 .004-33.485m57.991 26.557-16.996-9.814 17-9.815m-58 26.557 57.999-33.485M.001 133.742l-.004-66.8m29.001 50.23-57.99-33.485m45.994-6.928-5.005 2.89V73.87m11.005 6.353L5.999 90.04l-.002-19.633M29 103.317l-5-2.887 5.002-2.887m28.99-30.6L28.993 83.687zm.011.006 28.993 16.738-.004 33.485m0 0-28.997 16.57m0 0-28.997-16.57m0 0 .004-33.485m22.99-13.28v19.627l-16.995-9.813m-5.999 36.95 57.998-33.484m-29 50.055-.005-66.8m29.001 50.23-57.99-33.485M29 103.314l5-2.886-5-2.886m11.996-20.786 4.996 2.885v-5.77m-16.994 36.373 17-9.815L29 90.615M57.998 66.94l-.003-33.484zm-.012.008-28.992 16.74L-.002 66.94l.148-33.397 28.849-16.827L57.99 33.463M.084 47.363 4.997 50.2.06 53.05m5.936 17.356 16.998-9.812v19.63m35.003-20.212L41 50.2l16.996-9.812m-57.878.067 16.88 9.745L.03 59.996m28.966-43.28v66.971M.144 33.544 57.999 66.94m-58 .001L57.99 33.463M40.994 76.759v-5.78l5.004 2.89m-5.004-50.221v5.772l5-2.886m-11 53.689V60.589l17.004 9.815m-40.003 3.467 5-2.887v5.775m41.002-29.444L53 50.2l4.998 2.885M22.995 20.217v19.589l-16.88-9.744m5.97-3.481 4.91 2.835v-5.7m18-3.535v19.63l16.997-9.813"/></pattern>'
        patId = "anger"
    elif getEmotions()[0][0] == "disgust":
        patternString = f'<pattern id="disgust" width="15.825" height="26.667" patternTransform="rotate(90)scale(5)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="{accentRGB()}" d="M-3.176 15.632a1.5 1.5 0 0 0-.294.038 1.463 1.463 0 0 0-1.08 1.754l.013.05c.503 2.134 1.828 3.999 3.533 5.201a9.2 9.2 0 0 0 5.803 1.68c2.012-.098 3.962-.883 5.422-2.17a8.1 8.1 0 0 0 1.93-2.494 9 9 0 0 0 2.67 2.984 9.2 9.2 0 0 0 5.803 1.68c2.012-.098 3.962-.883 5.422-2.17 1.472-1.277 2.454-3.068 2.7-4.944a.22.22 0 0 0-.16-.234c-.11-.036-.221.037-.246.148a7.3 7.3 0 0 1-2.932 4.207 7.6 7.6 0 0 1-4.772 1.325c-1.656-.098-3.227-.76-4.392-1.815-1.178-1.043-1.938-2.478-2.098-3.95a.4.4 0 0 0-.036-.172 1.463 1.463 0 0 0-1.755-1.08 1.463 1.463 0 0 0-1.079 1.754l.012.05q.183.769.5 1.484a7.35 7.35 0 0 1-2.205 2.404 7.6 7.6 0 0 1-4.772 1.325c-1.656-.098-3.227-.76-4.392-1.815-1.178-1.043-1.938-2.478-2.098-3.95a.4.4 0 0 0-.036-.172 1.464 1.464 0 0 0-1.461-1.118M-11.51 2.298a1.463 1.463 0 0 0-1.373 1.792l.013.05c.503 2.135 1.828 4 3.533 5.202a9.2 9.2 0 0 0 5.802 1.68c2.012-.098 3.962-.883 5.422-2.171a8.1 8.1 0 0 0 1.931-2.493 9 9 0 0 0 2.67 2.983 9.2 9.2 0 0 0 5.802 1.68c2.012-.097 3.962-.882 5.422-2.17 1.473-1.276 2.454-3.067 2.7-4.944a.22.22 0 0 0-.16-.233c-.11-.037-.22.037-.245.147a7.3 7.3 0 0 1-2.933 4.208 7.6 7.6 0 0 1-4.771 1.325c-1.656-.098-3.227-.76-4.392-1.816-1.178-1.043-1.939-2.478-2.098-3.95a.4.4 0 0 0-.037-.172 1.463 1.463 0 0 0-1.754-1.08 1.463 1.463 0 0 0-1.08 1.755l.013.05c.12.512.29 1.007.5 1.483A7.35 7.35 0 0 1 1.25 8.03a7.6 7.6 0 0 1-4.773 1.325c-1.656-.098-3.226-.76-4.392-1.816-1.177-1.043-1.938-2.478-2.097-3.95a.4.4 0 0 0-.037-.172 1.464 1.464 0 0 0-1.46-1.118z"/></pattern>'
        patId = "disgust"
    elif getEmotions()[0][0] == "anticipation":
        patternString = f'<pattern id="anticipation" width="20" height="20" patternTransform="scale(2)" patternUnits="userSpaceOnUse"><rect width="100%" height="100%" fill="{fillColor}"/><path fill="none" stroke="{accentRGB()}" stroke-linecap="square" stroke-width="2.5" d="M3.25 10h13.5M10 3.25v13.5"/></pattern>'
        patId = "anticipation"
    return patternString, patId




def makeSVG():
    freq = text_object.affect_frequencies
    #more smiley = more positive sentiment
    positive = freq.get("positive", 0)
    negative = freq.get("negative", 0)
    mood_score = positive - negative
    controlY = 250 - (mood_score * 200)
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400">
<defs>
{selectPattern()[0]}
 </defs>
  <rect width="300" height="400" {maskCurvature()} fill="url(#{selectPattern()[1]})"/>
<path d="M 50 250 Q 150 {controlY} 250 250" fill="white" stroke="white" stroke-width="2" stroke-linecap="round" />
<rect x="0%" y="{25 + facialSymmetry()}%" width="140" height="{60 + createEyes()[1]}" {maskCurvature()} fill="rgb({complementary()[0]},{complementary()[1]}, {complementary()[2]})" />
<rect x="55%" y="{25 - facialSymmetry()}%" width="140" height="{60 + createEyes()[1]}" {maskCurvature()} fill="rgb({complementary()[0]},{complementary()[1]}, {complementary()[2]})"/>

<ellipse rx="{createEyes()[0]}" ry="{createEyes()[1]}" cx="25%" cy="{35 + facialSymmetry()}%" fill="white" stroke="white" stroke-width="2"/>
<ellipse rx="{createEyes()[0]}" ry="{createEyes()[1]}" cx="75%" cy="{35 - facialSymmetry()}%" fill="white" stroke="white" stroke-width="2"/>
</svg>
    '''
    return svg_content

#viewable
for file_dir in directory.iterdir():
    if file_dir.is_dir():
        print({file_dir.name})
        text_object.load_raw_text(getInternalResponses(file_dir))
        freqSort = sorted(text_object.affect_frequencies.items(), key=lambda x: x[1], reverse=True)
        print(f"Overall vibe: {getVibe()}")
        print(f"Formatted Top 3: {getEmotions()}")
        print(f"Emotions with percentages: {topEmotionPercentage()}")
        print(f"Raw Scores: {text_object.raw_emotion_scores}")
        with open(f"native_image_{file_dir.name}.svg", "w") as file:
            file.write(makeSVG())
        print("SVG generated successfully.")
        print(complementary())

