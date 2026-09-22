module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: [
          "python install.py"
        ],
        on: [{
          event: "/setup complete/i",
          done: true
        }]
      }
    }
  ]
}