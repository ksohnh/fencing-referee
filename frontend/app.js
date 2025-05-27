// 1) Top-level log so we know app.js ran
console.log("📦 app.js loaded");

const api = "http://localhost:8000";

async function init() {
  console.log("🚀 init running");
  const sel = document.getElementById("video-select");

  // 2) Fetch and filter only .mp4 files
  let vids = await fetch(`${api}/videos`)
    .then(r => r.json())
    .catch(e => { console.error("Failed to fetch /videos:", e); return []; });

  vids = vids.filter(v => v.toLowerCase().endsWith(".mp4"));
  if (vids.length === 0) {
    sel.innerHTML = '<option disabled>No videos uploaded</option>';
    return;
  }

  // Populate dropdown
  sel.innerHTML = "";  
  vids.forEach(v => sel.add(new Option(v, v)));

  // 3) Auto-select & load the first video
  sel.value = vids[0];
  loadVideo(vids[0]);

  // When user picks another
  sel.onchange = () => loadVideo(sel.value);
}

window.addEventListener("load", init);

function loadVideo(filename) {
  console.log("🎬 loadVideo()", filename);

  const container = document.getElementById("stage-container");
  console.log(" — container before clear:", container.innerHTML);

  container.innerHTML = "";      // clear old
  console.log(" — container after clear:", container.innerHTML);

  // Create & append the <video> element
  const video = document.createElement("video");
  video.src      = `${api}/videos/${filename}`;
  video.controls = true;
  video.width    = 800;
  video.height   = 450;

  // Listen for load success or failure
  video.addEventListener("loadeddata", () => {
    console.log("✅ video loaded, duration:", video.duration);
  });
  video.addEventListener("error", (e) => {
    console.error("❌ video failed to load:", video.src, e);
  });

  container.appendChild(video);
  console.log(" — container after append:", container.innerHTML);

  const konvaStageElement = document.createElement("div");
  konvaStageElement.id = "konva-stage";
  container.appendChild(konvaStageElement);

  // Konva overlay
  const stage = new Konva.Stage({
    container: "konva-stage",
    width: 800,
    height: 450
  });
  const layer = new Konva.Layer();
  stage.add(layer);

  stage.on("click", () => {
    const pos = stage.getPointerPosition();
    console.log("🔴 click at", pos);
    const circle = new Konva.Circle({ x:pos.x, y:pos.y, radius:8, stroke:"red" });
    layer.add(circle);
    layer.draw();
  });
}



// function loadVideo(filename) {
//   console.log("🎬 loadVideo()", filename);

//   const container = document.getElementById("stage-container");
//   container.innerHTML = "";      // clear old

//   // 4a) Create & append the <video> element
//   const video = document.createElement("video");
//   video.src      = `${api}/videos/${filename}`;
//   video.controls = true;
//   video.width    = 800;
//   video.height   = 450;
//   container.appendChild(video);

//   video.addEventListener("loadeddata", () => {
//     console.log("✅ video loaded, duration:", video.duration);
//   });

//   // 4b) Now overlay Konva on top
//   const stage = new Konva.Stage({
//     container: "stage-container",
//     width: 800,
//     height: 450
//   });
//   const layer = new Konva.Layer();
//   stage.add(layer);

//   stage.on("click", () => {
//     console.log("🔴 click at", stage.getPointerPosition());
//     const pos = stage.getPointerPosition();
//     const circle = new Konva.Circle({
//       x: pos.x,
//       y: pos.y,
//       radius: 8,
//       stroke: "red"
//     });
//     layer.add(circle);
//     layer.draw();
//   });
// }
