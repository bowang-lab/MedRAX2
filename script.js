document.addEventListener('DOMContentLoaded', () => {
    $("#demoVideo").on("canplay", function() {
      this.playbackRate = 1.5;
    });
    const toggleButton = document.getElementById("toggleButton");
    const root = document.documentElement;

    // Check if user's preference is stored
    const darkModeEnabled = localStorage.getItem("darkModeEnabled");

    // Set the initial mode based on user's preference
    if (darkModeEnabled === "true") {
        root.classList.add("darkmode");
        toggleButton.classList.toggle("inactive");
        toggleButton.classList.toggle("active");
        toggleButton.textContent = "Light Mode";
    }

    toggleButton.addEventListener("click", () => {
        root.classList.toggle("darkmode");
        toggleButton.classList.toggle("active");
        toggleButton.classList.toggle("inactive");

        // Store user's preference in localStorage
        const isDarkMode = root.classList.contains("darkmode");
        localStorage.setItem("darkModeEnabled", isDarkMode);

        if (toggleButton.classList.contains("active")) {
            toggleButton.textContent = "Light Mode";
          } else {
            toggleButton.textContent = "Dark Mode";
        }
      });

      // Define scroll links
      const scrollLinks = document.querySelectorAll('.scroll-link');

      scrollLinks.forEach(function (link) {
        link.addEventListener('click', function (e) {
          e.preventDefault();
          const targetId = this.getAttribute('href').substring(1); // Remove the "#" character
          const targetSection = document.getElementById(targetId);
    
          if (targetSection) {
            targetSection.scrollIntoView({
              behavior: 'instant'
            });
          }
        });
      });
    
    // Menu/navigation controls
    const navigationButton = document.getElementById("jumpTo");
    const navigation = document.getElementById("navigation");

    navigationButton.addEventListener("mouseenter", (event) => {
        navigation.style.display = 'flex';
    });

    navigation.addEventListener("mouseleave", (event) => {
        navigation.style.display = 'none';
    });

    document.addEventListener('mouseout', function(event) {
        if (event.relatedTarget === null) {
            navigation.style.display = 'none';
        }
    });

    const originalSection = document.getElementById("navigation");
    const clonedContent = originalSection.cloneNode(true);
    
    clonedContent.style.display = "flex";
});
