from app import create_app  # Import the create_app function from the app package

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
