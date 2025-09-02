/**
 * Advanced PDF Export for Comics
 * Converts edited comic to PDF using jsPDF or html2canvas
 */

// Option 1: Using browser's print-to-PDF (already implemented)
// This is the most reliable method

// Option 2: Client-side PDF generation (requires libraries)
function exportToPDFAdvanced() {
    // This would require including jsPDF and html2canvas libraries
    // Example implementation:
    
    /*
    // Include these libraries in your HTML:
    // <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    // <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'mm', 'a4');
    
    // Hide controls
    document.querySelector('.edit-controls').style.display = 'none';
    
    // Get all comic pages
    const pages = document.querySelectorAll('.comic-page');
    let currentPage = 0;
    
    function processPage() {
        if (currentPage >= pages.length) {
            // Save PDF
            pdf.save('comic-edited.pdf');
            document.querySelector('.edit-controls').style.display = 'block';
            return;
        }
        
        html2canvas(pages[currentPage], {
            scale: 2,
            useCORS: true,
            logging: false
        }).then(canvas => {
            const imgData = canvas.toDataURL('image/png');
            
            if (currentPage > 0) {
                pdf.addPage();
            }
            
            // Calculate dimensions to fit A4
            const pdfWidth = 210;
            const pdfHeight = 297;
            const imgWidth = canvas.width;
            const imgHeight = canvas.height;
            const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
            
            const finalWidth = imgWidth * ratio;
            const finalHeight = imgHeight * ratio;
            const x = (pdfWidth - finalWidth) / 2;
            const y = (pdfHeight - finalHeight) / 2;
            
            pdf.addImage(imgData, 'PNG', x, y, finalWidth, finalHeight);
            
            currentPage++;
            processPage();
        });
    }
    
    processPage();
    */
}

// Option 3: Server-side PDF generation
function requestServerPDF() {
    // Collect all edited data
    const editedData = {
        bubbles: []
    };
    
    document.querySelectorAll('.speech-bubble').forEach((bubble, index) => {
        editedData.bubbles.push({
            index: index,
            text: bubble.innerText,
            left: bubble.style.left,
            top: bubble.style.top,
            width: bubble.offsetWidth,
            height: bubble.offsetHeight
        });
    });
    
    // Send to server for PDF generation
    fetch('/generate-pdf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(editedData)
    })
    .then(response => response.blob())
    .then(blob => {
        // Download the PDF
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'comic-edited.pdf';
        a.click();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        console.error('PDF generation failed:', error);
        alert('PDF generation failed. Please use the Print option instead.');
    });
}