# 🎮 Unity Comic Integration Guide

## For Unity, You Have Two Options:

### Option 1: Individual Panels (Recommended)
Export each panel as **400×540** without borders

**Benefits:**
- ✅ More flexible in Unity
- ✅ Can animate panels individually
- ✅ Easy to rearrange
- ✅ Better performance (smaller textures)

### Option 2: Full Pages
Export complete pages as **800×1080** with or without borders

**Benefits:**
- ✅ Simpler to implement
- ✅ One texture per page
- ✅ Maintains exact layout

## Do You Need Borders?

**Short answer: NO** - Unity doesn't need borders

### Without Borders (Recommended):
- Clean images
- You can add borders in Unity with UI system
- More flexible styling options
- Smaller file sizes

### With Borders:
- Use only if you want that specific look
- Borders become part of the image
- Can't change border style later

## Unity Setup Guide

### 1. **Texture Import Settings**
```
Texture Type: Sprite (2D and UI)
Pixels Per Unit: 100
Filter Mode: Bilinear
Max Size: 2048
Format: RGBA 32 bit
```

### 2. **For 400×540 Panels**
```csharp
// Create 2x2 grid in Unity
float panelWidth = 400f;
float panelHeight = 540f;

// Position panels
panel1.position = new Vector3(0, 0, 0);
panel2.position = new Vector3(400, 0, 0);
panel3.position = new Vector3(0, -540, 0);
panel4.position = new Vector3(400, -540, 0);
```

### 3. **For 800×1080 Pages**
```csharp
// Simple page display
GameObject comicPage = new GameObject("ComicPage");
Image pageImage = comicPage.AddComponent<Image>();
pageImage.sprite = yourPageSprite;

// Set size
RectTransform rt = comicPage.GetComponent<RectTransform>();
rt.sizeDelta = new Vector2(800, 1080);
```

## Preparing Images for Unity

### Remove Borders (CSS):
```css
.panel { 
    border: none !important; 
}
.comic-grid { 
    border: none !important; 
}
```

### Export Options:

1. **Individual Panels** (400×540 each)
   - No borders
   - Transparent or white background
   - PNG format

2. **Full Pages** (800×1080 each)
   - No borders (add in Unity)
   - White background
   - PNG or JPEG

## Unity Comic Viewer Example

```csharp
public class ComicViewer : MonoBehaviour
{
    public Sprite[] comicPages; // 800x1080 pages
    public Image displayImage;
    
    private int currentPage = 0;
    
    void Start()
    {
        ShowPage(0);
    }
    
    public void NextPage()
    {
        currentPage++;
        if (currentPage < comicPages.Length)
            ShowPage(currentPage);
    }
    
    void ShowPage(int pageIndex)
    {
        displayImage.sprite = comicPages[pageIndex];
    }
}
```

## Best Practices for Unity

1. **Use Power of 2 textures** when possible (512, 1024, 2048)
2. **Compress textures** in Unity import settings
3. **Use UI Canvas** for comic display
4. **Consider mobile** - 800×1080 is perfect for portrait mode

## Size Compatibility

Unity handles any size, but consider:
- **Mobile**: 800×1080 works great
- **Desktop**: May need scaling
- **Memory**: Each 800×1080 = ~3.5MB uncompressed

## Recommended Workflow

1. Export from comic system **without borders**
2. Import to Unity as **Sprites**
3. Use **Canvas UI** system
4. Add borders/effects in Unity
5. Scale with **Canvas Scaler** for different devices

## Result

- No borders needed in images
- Unity makes size compatible automatically
- More flexibility without baked-in borders
- Professional comic viewer in Unity!