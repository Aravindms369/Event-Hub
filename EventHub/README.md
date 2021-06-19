# Event-Hub

4. Apply migrations:
		
		python manage.py migrate
		python manage.py loaddata fixtures/init_data.json
		
5. Run the server

		python manage.py runserver
		
6. Run chat server in separate terminal
        
        python manage.py run_chat_server
7. Access from the browser at `http://127.0.0.1:8000`