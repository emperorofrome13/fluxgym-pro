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
          "python -m venv .venv || py -3 -m venv .venv",
          ".venv\\Scripts\\python.exe -m pip install --upgrade pip",
          ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ]
      }
    }
  ]
}
