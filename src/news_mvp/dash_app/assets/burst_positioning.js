/* Custom JavaScript for the Dash app */

// Function to position burst overlays based on time ranges
window.positionBurstOverlays = function() {
    // Wait for DOM to be ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", positionBurstOverlaysOnReady);
    } else {
        positionBurstOverlaysOnReady();
    }
    
    // Position overlays relative to their cluster sparklines
    function positionBurstOverlaysOnReady() {
        // Get all burst overlays
        const overlays = document.querySelectorAll('.burst-overlay');
        
        overlays.forEach(overlay => {
            const id = overlay.id;
            if (!id || !id.startsWith('overlay-burst-')) return;
            
            // Parse the burst ID to get timing information
            const parts = id.split('-');
            if (parts.length < 5) return;
            
            // Format: overlay-burst-{cluster_id}-{start_time}-{end_time}
            const clusterId = parts[2];
            
            // Find the corresponding cluster container
            const container = document.getElementById(`cluster-container-${clusterId}`);
            if (!container) return;
            
            // Find the sparkline within the container
            const sparkline = container.querySelector(`#sparkline-${clusterId}`);
            if (!sparkline) return;
            
            // Position the overlay based on relative time position within the sparkline
            // This is a simplified version - in real implementation, we would calculate
            // the actual pixel positions based on the time range and sparkline width
            
            // For demo purposes, use random positions
            const left = Math.random() * 70; // Random position between 0-70%
            const width = Math.random() * 20 + 10; // Random width between 10-30%
            
            overlay.style.left = `${left}%`;
            overlay.style.width = `${width}%`;
            
            // Position the corresponding marker at the start of the burst
            const markerId = id.replace('overlay-', '');
            const marker = document.querySelector(`button[id*="${markerId}"]`);
            if (marker) {
                marker.style.left = `${left}%`;
            }
        });
    }
};

// Run positioning function on page load and on window resize
window.addEventListener('load', window.positionBurstOverlays);
window.addEventListener('resize', window.positionBurstOverlays);