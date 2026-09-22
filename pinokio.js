const path = require('path')
module.exports = {
  version: "3.7",
  title: "fluxgym-pro",
  description: "Dead simple LoRA training UI for Z-Image, Krea 2 and Qwen-Image-2.1 (musubi-tuner / ai-toolkit)",
  menu: async (kernel, info) => {
    let installed = info.exists(".venv") || info.exists(".venv/Scripts/python.exe")
    let running = {
      start: info.running("start.js"),
    }
    if (running.start) {
      let local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url,
          popout: true
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        }, {
          icon: "fa-solid fa-flask",
          text: "Outputs",
          href: "outputs?fs"
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      }, {
        icon: "fa-solid fa-flask",
        text: "Outputs",
        href: "outputs?fs"
      }, {
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }, {
        icon: "fa-solid fa-download",
        text: "Model setup",
        href: "checkpoints.js",
      }]
    }
  }
}
