/* ==========================================
   CITYVISION AI DASHBOARD
   FRONTEND DEMO
========================================== */


/* CLOCK */

function updateClock() {

    const clock = document.getElementById("clock");

    const now = new Date();

    const time = now.toLocaleTimeString(
        "en-IN",
        {
            hour12: false
        }
    );

    clock.textContent = time;
}


setInterval(updateClock, 1000);

updateClock();



/* PAGE NAVIGATION */

const menuItems =
    document.querySelectorAll(".menu-item[data-page]");

const pages =
    document.querySelectorAll(".page");


const pageTitles = {

    dashboard: {
        title: "City Traffic Dashboard",
        subtitle: "Real-time AI-powered traffic intelligence"
    },

    cameras: {
        title: "Camera Network",
        subtitle: "Monitor connected ANPR camera nodes"
    },

    trajectory: {
        title: "Trajectory Tracking",
        subtitle: "Spatial-temporal vehicle movement analysis"
    },

    analytics: {
        title: "Traffic Analytics",
        subtitle: "City-wide traffic flow and movement intelligence"
    },

    alerts: {
        title: "Alert Center",
        subtitle: "Monitor system and traffic events"
    }

};


function showPage(pageId) {

    pages.forEach(page => {

        page.classList.remove("active-page");

    });


    const selectedPage =
        document.getElementById(pageId);


    if (selectedPage) {

        selectedPage.classList.add("active-page");

    }


    menuItems.forEach(item => {

        item.classList.remove("active");

        if (item.dataset.page === pageId) {

            item.classList.add("active");

        }

    });


    if (pageTitles[pageId]) {

        document.getElementById("pageTitle")
            .textContent = pageTitles[pageId].title;

        document.getElementById("pageSubtitle")
            .textContent = pageTitles[pageId].subtitle;

    }


    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

}


menuItems.forEach(item => {

    item.addEventListener("click", function(event) {

        event.preventDefault();

        showPage(this.dataset.page);

    });

});



/* MOBILE SIDEBAR */

const mobileMenu =
    document.querySelector(".mobile-menu");

const sidebar =
    document.querySelector(".sidebar");


mobileMenu.addEventListener("click", () => {

    sidebar.classList.toggle("mobile-open");

});


menuItems.forEach(item => {

    item.addEventListener("click", () => {

        sidebar.classList.remove("mobile-open");

    });

});



/* VEHICLE SEARCH */

function searchVehicle() {

    const input =
        document.getElementById("plateInput");

    const plate =
        input.value.trim().toUpperCase();


    if (plate === "") {

        alert(
            "Please enter a demo vehicle identifier."
        );

        return;

    }


    const plateDisplay =
        document.querySelector(".plate-display");

    plateDisplay.textContent = plate;


    document.getElementById("trajectoryResult")
        .scrollIntoView({
            behavior: "smooth"
        });

}



/* TRAFFIC CHART */

const trafficCanvas =
    document.getElementById("trafficChart");


if (trafficCanvas) {

    new Chart(
        trafficCanvas,
        {

            type: "line",

            data: {

                labels: [
                    "00:00",
                    "02:00",
                    "04:00",
                    "06:00",
                    "08:00",
                    "10:00",
                    "12:00",
                    "14:00",
                    "16:00",
                    "18:00",
                    "20:00",
                    "22:00"
                ],

                datasets: [

                    {
                        label: "Vehicles",

                        data: [
                            720,
                            540,
                            390,
                            680,
                            1850,
                            2340,
                            2180,
                            2010,
                            2390,
                            2850,
                            2480,
                            1720
                        ],

                        borderWidth: 2,

                        tension: .4,

                        fill: true,

                        backgroundColor:
                            "rgba(59,130,246,.08)",

                        borderColor:
                            "#3b82f6",

                        pointRadius: 2,

                        pointBackgroundColor:
                            "#60a5fa"

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        display: false
                    }

                },

                scales: {

                    x: {

                        grid: {
                            display: false
                        },

                        ticks: {
                            color: "#667085",
                            font: {
                                size: 8
                            }
                        }

                    },

                    y: {

                        grid: {
                            color: "#1b2431"
                        },

                        ticks: {
                            color: "#667085",
                            font: {
                                size: 8
                            }
                        }

                    }

                }

            }

        }

    );

}



/* ANALYTICS CHART */

const analyticsCanvas =
    document.getElementById("analyticsChart");


if (analyticsCanvas) {

    new Chart(
        analyticsCanvas,
        {

            type: "bar",

            data: {

                labels: [
                    "00:00",
                    "02:00",
                    "04:00",
                    "06:00",
                    "08:00",
                    "10:00",
                    "12:00",
                    "14:00",
                    "16:00",
                    "18:00",
                    "20:00",
                    "22:00"
                ],

                datasets: [

                    {
                        label: "Vehicle Flow",

                        data: [
                            520,
                            410,
                            360,
                            740,
                            1800,
                            2380,
                            2140,
                            2070,
                            2450,
                            2910,
                            2570,
                            1690
                        ],

                        borderRadius: 5,

                        backgroundColor:
                            "#3b82f6"

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        display: false
                    }

                },

                scales: {

                    x: {

                        grid: {
                            display: false
                        },

                        ticks: {
                            color: "#667085"
                        }

                    },

                    y: {

                        grid: {
                            color: "#1b2431"
                        },

                        ticks: {
                            color: "#667085"
                        }

                    }

                }

            }

        }

    );

}



/* SIMULATED LIVE VEHICLE COUNTER */

let vehicles = 24681;


setInterval(() => {

    vehicles +=
        Math.floor(
            Math.random() * 7
        ) + 1;


    const counter =
        document.getElementById("vehicleCount");


    if (counter) {

        counter.textContent =
            vehicles.toLocaleString();

    }

}, 3000);



/* MAP CAMERA HOVER EFFECT */

const cameraMarkers =
    document.querySelectorAll(".camera-marker");


cameraMarkers.forEach(marker => {

    marker.addEventListener(
        "mouseenter",
        () => {

            marker.style.transform =
                "scale(1.3)";

        }
    );


    marker.addEventListener(
        "mouseleave",
        () => {

            marker.style.transform =
                "scale(1)";

        }
    );

});