# 🎭 Smart Comic Generation with Emotion Matching

The Flask app now includes smart comic generation that matches facial expressions with dialogue and creates 10-15 panel story summaries!

## 🚀 How to Use

### 1. Start the Flask App
```bash
python app_enhanced.py
```

### 2. Open in Browser
Navigate to `http://localhost:5000`

### 3. Upload Video or Paste Link
- Click the upload button to select a video file
- OR click the link button to paste a YouTube URL

### 4. Enable Smart Comic Options
You'll see two checkboxes:
- **☑️ Smart Mode**: Creates a 10-15 panel summary instead of full comic
- **☑️ Match facial expressions**: Matches character emotions with dialogue

### 5. Click Submit
The app will:
1. Extract audio and generate real subtitles
2. Analyze the story structure
3. Identify key moments (intro, conflict, climax, resolution)
4. Match facial expressions with dialogue emotions
5. Create a condensed comic with emotion-styled speech bubbles

## 🎨 Features

### Smart Story Summarization
- Automatically identifies key story moments
- Reduces hours of video to 10-15 essential panels
- Maintains narrative flow and coherence
- Prioritizes emotional peaks and turning points

### Emotion Matching
- Analyzes facial expressions in each frame
- Analyzes emotions in dialogue text
- Finds frames where face matches dialogue mood
- Styles speech bubbles based on emotion:
  - 😊 Happy: Green border, bouncing animation
  - 😢 Sad: Blue border, drooping effect
  - 😠 Angry: Red jagged border, larger text
  - 😲 Surprised: Orange burst shape
  - 😐 Neutral: Standard black border

### Intelligent Panel Selection
- Always includes introduction and conclusion
- Finds story turning points (but, however, suddenly)
- Identifies emotional peaks
- Detects action moments
- Ensures even distribution across story

## 📁 Output Files

After generation, you'll find:
- `output/page.html` - Regular comic (all panels)
- `output/smart_comic_viewer.html` - Smart comic summary
- `output/emotion_comic.json` - Comic data with emotion analysis

## 🎯 Example Results

**Input**: 30-minute video with 500+ subtitles
**Output**: 12-panel comic showing:
- Opening scene with character introduction
- First conflict moment
- Rising tension scenes
- Climactic confrontation
- Resolution and ending

Each panel has:
- Carefully selected frame matching the dialogue emotion
- Emotion-styled speech bubble
- Key dialogue that drives the story forward

## 🛠️ Technical Details

The smart comic generation uses:
- **Facial Expression Analysis**: OpenCV cascades for face/smile detection
- **Text Emotion Analysis**: Keyword and punctuation analysis
- **Story Structure Detection**: Identifies narrative phases
- **Importance Scoring**: Rates each moment's significance
- **Emotion Matching**: Calculates match scores between face and text

## 💡 Tips

1. **For Best Results**:
   - Use videos with clear dialogue
   - Ensure faces are visible in most scenes
   - Videos with emotional variety work best

2. **Customization**:
   - Uncheck "Smart Mode" for full comic
   - Uncheck "Match expressions" for faster processing
   - Both options can be used independently

3. **Performance**:
   - Smart mode is faster (fewer panels to process)
   - Emotion matching adds ~10-15 seconds
   - Total time: 2-5 minutes for most videos

## 🎉 Benefits

- **Time Saving**: Get the story essence without reading hundreds of panels
- **Better Storytelling**: Key moments are preserved and highlighted
- **Emotional Consistency**: Faces match the dialogue mood
- **Visual Impact**: Emotion styling makes comics more expressive
- **Automated**: No manual selection or editing needed

The smart comic feature transforms long videos into concise, emotionally-resonant visual stories!