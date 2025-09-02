# 📄 Comic Page Images (800x1080)

## Overview

The system now automatically generates individual page images at **800x1080 resolution** for each comic page. This makes it easy to:
- Share comic pages on social media
- Create printable versions
- Use pages in other applications
- Archive your comics

## Features

### 🎨 What You Get

1. **Individual Page Files**
   - Each comic page saved as a separate PNG image
   - Resolution: 800x1080 pixels (portrait)
   - High quality with 95% compression
   - Numbered sequentially (page_001.png, page_002.png, etc.)

2. **Complete Comic Layout**
   - 2x2 panel grid preserved
   - Speech bubbles included
   - Black borders around panels
   - Page numbers at bottom

3. **Gallery Viewer**
   - HTML gallery to view all pages
   - Download individual pages
   - Download all pages at once
   - Thumbnail preview

## How It Works

### Automatic Generation

After creating a comic:
1. System generates the interactive HTML comic
2. Extracts individual panels (640x800)
3. **Creates page images (800x1080)** ← NEW!
4. Saves to `output/page_images/`

### Access Page Images

**Option 1: From Comic Viewer**
- Click the **"🖼️ View Page Images"** button
- Opens gallery in new tab

**Option 2: Direct Access**
- Navigate to: `output/page_images/index.html`
- Or go to: http://localhost:5000/output/page_images/index.html

**Option 3: File System**
- Find images in: `/workspace/output/page_images/`
- Files: `page_001.png`, `page_002.png`, etc.

## Page Image Layout

Each 800x1080 image contains:

```
┌─────────────────────┐
│   Comic Page X      │ 800px
├──────────┬──────────┤
│          │          │
│  Panel 1 │  Panel 2 │
│          │          │
├──────────┼──────────┤ 1080px
│          │          │
│  Panel 3 │  Panel 4 │
│          │          │
├─────────────────────┤
│      Page X         │
└─────────────────────┘
```

## Use Cases

### 1. **Social Media Sharing**
- Perfect size for Instagram stories (9:16 ratio)
- Easy to share individual pages
- Maintains readability on mobile

### 2. **Printing**
- Standard portrait orientation
- Good resolution for A4/Letter printing
- Multiple pages per sheet possible

### 3. **Digital Publishing**
- Ready for e-book conversion
- Suitable for web comics
- Easy to create PDFs

### 4. **Archiving**
- Consistent file naming
- Portable format
- No dependencies needed

## Technical Details

### Image Specifications
- **Format**: PNG
- **Resolution**: 800x1080 pixels
- **Color**: RGB (full color)
- **Compression**: 95% quality
- **File size**: ~200-500KB per page

### Processing Steps
1. Load comic page data from JSON
2. Create white canvas (800x1080)
3. Draw 2x2 panel grid with borders
4. Place panel images (maintaining aspect ratio)
5. Add speech bubbles with text
6. Add page number
7. Save as PNG

## Gallery Features

The page image gallery includes:

- **Grid View**: See all pages at once
- **Download Links**: Individual page downloads
- **Batch Download**: Get all pages with one click
- **Responsive Design**: Works on mobile/tablet
- **Quick Preview**: Hover to enlarge

## Tips

### For Best Results
1. **Original Quality**: Use high-quality video/images
2. **Clear Text**: Ensure subtitles are readable
3. **Good Lighting**: Better source = better pages

### Customization
- Edit `backend/page_image_generator.py` to:
  - Change resolution (default 800x1080)
  - Adjust border thickness
  - Modify panel spacing
  - Change background color

### Storage
- Page images are saved in: `output/page_images/`
- Each comic generation overwrites previous pages
- Consider backing up pages you want to keep

## Example Usage

### After Comic Generation
```
✅ Comic generation completed in 2.41 minutes
📄 Generating page images (800x1080)...
📄 Generated page 1/12: page_001.png
📄 Generated page 2/12: page_002.png
...
📄 Generated page 12/12: page_012.png
📋 Page index created: output/page_images/index.html
✅ Generated 12 page images (800x1080)
📄 Page gallery available at: output/page_images/index.html
```

### Accessing Images
1. Click "🖼️ View Page Images" in comic viewer
2. Browse gallery
3. Download individual pages or all at once

## Future Enhancements

Possible additions:
- Custom resolutions (720x1280, 1080x1920)
- Different layouts (3x3, 1x4)
- Watermark options
- JPEG format support
- Automatic upload to cloud

Enjoy your page images! 🎨📄