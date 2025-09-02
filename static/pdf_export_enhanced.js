/**
 * Enhanced PDF Export Settings Guide
 * 
 * For best results when exporting to PDF:
 */

// Browser-specific settings for optimal PDF export

const PDFExportGuide = {
    
    // Chrome/Edge settings
    chrome: {
        destination: "Save as PDF",
        layout: "Landscape",
        paperSize: "A4",
        margins: "None",
        scale: "Fit to page width",
        backgroundGraphics: true,
        headers: false
    },
    
    // Firefox settings
    firefox: {
        destination: "Save as PDF",
        orientation: "Landscape",
        paperSize: "A4",
        margins: "None",
        scale: 100,
        printBackgrounds: true,
        shrinkToFit: true
    },
    
    // Safari settings
    safari: {
        PDF: "Save as PDF",
        orientation: "Landscape",
        paperSize: "A4",
        scale: "100%",
        printBackgrounds: true
    }
};

// Alternative CSS for better PDF sizing
const enhancedPrintCSS = `
@media print {
    /* Force exact dimensions */
    html, body {
        width: 297mm;
        height: 210mm;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    
    /* Hide everything except comic pages */
    body > *:not(.comic-container) {
        display: none !important;
    }
    
    /* Comic container full size */
    .comic-container {
        width: 297mm !important;
        height: 210mm !important;
        margin: 0 !important;
        padding: 0 !important;
        position: relative !important;
    }
    
    /* Each page exact A4 landscape */
    .comic-page {
        width: 297mm !important;
        height: 210mm !important;
        page-break-after: always !important;
        page-break-inside: avoid !important;
        position: relative !important;
        margin: 0 !important;
        padding: 10mm !important;
        box-sizing: border-box !important;
    }
    
    /* Grid fills available space */
    .comic-grid {
        width: 100% !important;
        height: 100% !important;
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        grid-template-rows: 1fr 1fr !important;
        gap: 5mm !important;
    }
    
    /* Panels fill grid cells */
    .panel {
        width: 100% !important;
        height: 100% !important;
        position: relative !important;
        overflow: hidden !important;
        border: 2px solid black !important;
    }
    
    .panel img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
    }
    
    /* Print settings */
    @page {
        size: A4 landscape;
        margin: 0;
    }
}
`;

// Function to prepare document for PDF export
function preparePDFExport() {
    // Remove any existing print styles
    const existingPrintStyles = document.querySelector('#pdf-print-styles');
    if (existingPrintStyles) {
        existingPrintStyles.remove();
    }
    
    // Add enhanced print styles
    const styleEl = document.createElement('style');
    styleEl.id = 'pdf-print-styles';
    styleEl.innerHTML = enhancedPrintCSS;
    document.head.appendChild(styleEl);
    
    // Temporarily modify layout for better printing
    const comicPages = document.querySelectorAll('.comic-page');
    comicPages.forEach(page => {
        page.style.pageBreakAfter = 'always';
        page.style.pageBreakInside = 'avoid';
    });
    
    // Show browser-specific instructions
    const userAgent = navigator.userAgent;
    let instructions = '';
    
    if (userAgent.includes('Chrome') || userAgent.includes('Edge')) {
        instructions = 'Chrome/Edge detected. Use these settings:\n' +
                      '• Layout: Landscape\n' +
                      '• Margins: None\n' +
                      '• Scale: Fit to page width';
    } else if (userAgent.includes('Firefox')) {
        instructions = 'Firefox detected. Use these settings:\n' +
                      '• Orientation: Landscape\n' +
                      '• Margins: None\n' +
                      '• Scale: 100%';
    } else if (userAgent.includes('Safari')) {
        instructions = 'Safari detected. Use these settings:\n' +
                      '• Orientation: Landscape\n' +
                      '• Scale: 100%';
    }
    
    if (instructions) {
        console.log('📄 PDF Export Settings:\n' + instructions);
    }
}

// Enhanced export function
function exportToPDFEnhanced() {
    preparePDFExport();
    
    setTimeout(() => {
        window.print();
    }, 100);
}

// Add to window for global access
window.exportToPDFEnhanced = exportToPDFEnhanced;