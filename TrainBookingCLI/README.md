# Train Booking Backend (Java)

A simple Java backend for train booking that runs from terminal.

## Requirements

- Java 17+
- Maven 3.8+

## Run

```bash
mvn clean compile exec:java
```

Server starts on `http://localhost:8080`.

## Endpoints

- `GET /health`
- `GET /trains`
- `GET /bookings`
- `POST /bookings`

## Example Requests

List trains:

```bash
curl http://localhost:8080/trains
```

Create booking:

```bash
curl -X POST http://localhost:8080/bookings \
  -H "Content-Type: application/json" \
  -d '{"trainId":"T101","passengerName":"Bimal","seats":2}'
```

List bookings:

```bash
curl http://localhost:8080/bookings
```
