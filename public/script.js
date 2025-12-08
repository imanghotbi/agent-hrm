window.addEventListener("load", function() {
    // Function to update placeholder
    function updatePlaceholder() {
        const textArea = document.querySelector('textarea');
        if (textArea) {
            textArea.placeholder = "پیام خود را اینجا بنویسید..."; // "Please write your message here..."
        }
    }

    // Run initially
    updatePlaceholder();

    // Run periodically in case the element is re-rendered by React
    setInterval(updatePlaceholder, 1000);
});