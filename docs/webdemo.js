var model;
var the_params;
var camera_enabled = false;
var flip_camera = false;
var camera_type = "environment";
var camera_img;
var current_img;
var target_FPS = 5;

async function load_model() {
  document.getElementById('load-model-div').innerText = 'Loading model...';
  // Create a simple model.
  const modelUrl =
     './vs_model_demo.tfjs/model.json';
  model = await tf.loadGraphModel(modelUrl);
  const params = await tf.util.fetch('./vs_model_demo.tfjs/params.json');
  const params_ = await params.json();
  the_params = tf.tensor(params_);

  document.getElementById('load-model-div').style = 'display: none';
  document.getElementById('select-stimulus').style = 'display: block';
}

async function updateDisplay() {
  if (camera_enabled) current_img = camera_img;

  // the array order is highly unpredictable, unfortunately. but it is stable once exported.
  var out = await model.execute({'image': current_img, 'parameters': the_params});
  var LL = out[0];
  var ML = out[1];
  var HL = out[2];
  var total = out[3];

  var standard_total = tf.tidy(() => {var min = tf.min(total); return tf.div(tf.sub(total, min), tf.sub(tf.max(total), min))});
  var standard_LL = tf.tidy(() => {var min = tf.min(LL); return tf.div(tf.sub(LL, min), tf.sub(tf.max(LL), min))});
  var standard_ML = tf.tidy(() => {var min = tf.min(ML); return tf.div(tf.sub(ML, min), tf.sub(tf.max(ML), min))});
  var standard_HL = tf.tidy(() => {var min = tf.min(HL); return tf.div(tf.sub(HL, min), tf.sub(tf.max(HL), min))});
  var combined_image = tf.tidy(() => {return tf.mul(current_img, standard_total.expandDims(2));});

  var stimulus = document.getElementById('stimulus');
  var output = document.getElementById('output');
  var LL_features = document.getElementById('LL_features');
  var ML_features = document.getElementById('ML_features');
  var HL_features = document.getElementById('HL_features');
  var combined = document.getElementById('combined');
  // update these as close to simultaneously as possible
  await Promise.all([tf.browser.toPixels(current_img, stimulus),
    tf.browser.toPixels(standard_LL, LL_features),
    tf.browser.toPixels(standard_ML, ML_features),
    tf.browser.toPixels(standard_HL, HL_features),
    tf.browser.toPixels(standard_total, output),
    tf.browser.toPixels(combined_image, combined)]);
  total.dispose();
  LL.dispose();
  ML.dispose();
  HL.dispose();
  standard_total.dispose();
  standard_LL.dispose();
  standard_ML.dispose();
  standard_HL.dispose();
  combined_image.dispose();
}

function disableCamera() {
  camera_enabled = false;
  document.getElementById('ffcam').disabled = false;
  document.getElementById('rfcam').disabled = false;
  document.getElementById('discam').disabled = true;
}
function enableCamera() {
  camera_enabled = true;
  if (current_img) current_img.dispose();
  document.getElementById('ffcam').disabled = true;
  document.getElementById('rfcam').disabled = true;
  document.getElementById('discam').disabled = false;
}
function enableFrontFacing() {
  camera_type = 'user';
  flip_camera = true;
  enableCamera();
}
function enableRearFacing() {
  camera_type = 'environment';
  flip_camera = false;
  enableCamera();
}
function setFPS(value) {
  target_FPS = Number(value);
}
function changeImage() {
  disableCamera();
  var imgbox = document.getElementById('imagebox');
  if (imgbox.files && imgbox.files[0]) {
    var img = document.getElementById('hiddenimg');
    img.src = URL.createObjectURL(imgbox.files[0]); // this is a "blob" url...
    img.onload = imageUpdated;
  }
}
async function imageUpdated() {
  var img_elt = document.getElementById('hiddenimg');
  if (current_img) current_img.dispose();
  current_img = tf.tidy(() => {
    var raw_img = tf.browser.fromPixels(img_elt);  // blob url
    raw_img = tf.div(raw_img, 255.);
    return tf.image.resizeBilinear(raw_img, [224,224]);
  });
  await updateDisplay();
}

async function cameraThread() {
  const videoElement = document.getElementById('hiddenvid');
  while (true) {
    while (!camera_enabled)
      await new Promise(resolve => setTimeout(resolve, 100));
    cam = await tf.data.webcam(videoElement, {'facingMode': camera_type});

    var frame_timer;
    var raw_img;
    while (camera_enabled) {
      frame_timer = new Promise(resolve => setTimeout(resolve, 1000/target_FPS));
      raw_img = await cam.capture();
      if (camera_img) camera_img.dispose();
      camera_img = tf.div(raw_img, 255.);
      if (flip_camera) {
        var unflipped_camera_img = camera_img;
        camera_img = tf.reverse(unflipped_camera_img, axis=1);
        unflipped_camera_img.dispose();
      }
      if (camera_img.shape[0] != 224 || camera_img.shape[1] != 224) {
        var unscaled_camera_img = camera_img;
        camera_img = tf.image.resizeBilinear(unscaled_camera_img, [224,224]);
        unscaled_camera_img.dispose();
      }
      raw_img.dispose();
      await updateDisplay();
      await frame_timer;
    }
    cam.stop();
  }
}

window.addEventListener('load', cameraThread);
