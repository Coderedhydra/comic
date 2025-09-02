# 📐 PDF Export Full Page Sizing Guide

## The Issue
When exporting to PDF, comic pages may appear small or not fill the entire page.

## Solutions

### 1. **Use Correct Print Settings** (Most Important!)

When you click "Export to PDF", use these settings in the print dialog:

#### **Recommended Settings:**
- **Destination**: Save as PDF
- **Layout**: **Landscape** ← Important!
- **Paper size**: A4 or Letter
- **Margins**: **None** or **Minimum**
- **Scale**: **Fit to page** or **100%**
- **Options**: ✓ Background graphics

### 2. **Browser-Specific Settings**

#### **Chrome/Edge:**
```
Layout: Landscape
Margins: None
Scale: Fit to page width
✓ Background graphics
```

#### **Firefox:**
```
Orientation: Landscape
Margins: None
Scale: Fit to Page
✓ Print backgrounds
```

#### **Safari:**
```
Orientation: Landscape
Scale: 100%
✓ Print backgrounds
```

### 3. **What I've Fixed**

The system now:
- Sets each comic page to fill the entire PDF page
- Uses landscape orientation for better fit
- Removes unnecessary margins
- Scales panels to maximum size
- Preserves 2x2 grid layout

### 4. **Manual Adjustments**

If pages still appear small:

1. **In Print Dialog:**
   - Change "Scale" to "Fit to page width"
   - Or set custom scale (try 120-150%)
   - Ensure margins are "None"

2. **Paper Size:**
   - Try "Letter" if A4 doesn't work well
   - Or use "Tabloid" for larger output

3. **Layout:**
   - Always use "Landscape" for comics
   - Portrait will make panels tiny

## Technical Details

The CSS now sets:
```css
/* Each comic page fills PDF page */
.comic-page {
    width: 100vw;
    height: 100vh;
    page-break-after: always;
}

/* Page settings */
@page {
    size: A4 landscape;
    margin: 10mm;
}
```

## Quick Checklist

Before clicking "Save" in print dialog:

- [ ] Layout = **Landscape**
- [ ] Margins = **None** or **Default**
- [ ] Scale = **Fit to page**
- [ ] Background graphics = **Enabled**
- [ ] Paper size = **A4** or **Letter**

## If Still Having Issues

### Option 1: Custom Scale
- Set Scale to "Custom"
- Try 115% or 125%
- This makes everything larger

### Option 2: Different Paper Size
- Try "Tabloid" (11x17)
- Gives more space for panels

### Option 3: Margins
- If "None" cuts off edges
- Use "Default" or "Narrow"

## Example Settings (Chrome)

1. Destination: **Save as PDF**
2. Pages: **All**
3. Layout: **Landscape**
4. Paper size: **A4**
5. Pages per sheet: **1**
6. Margins: **None**
7. Scale: **Default**
8. Options:
   - ✓ Background graphics
   - ✓ Selection only (unchecked)
   - ✓ Headers and footers (unchecked)

## Result

Your PDF should now have:
- Full-page comic panels
- No wasted white space
- Proper 2x2 grid layout
- All text and bubbles visible
- Professional appearance

Each comic page becomes one PDF page at maximum size!