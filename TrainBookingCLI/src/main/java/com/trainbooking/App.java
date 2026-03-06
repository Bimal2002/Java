package com.trainbooking;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

public class App {
    private static final Gson GSON = new Gson();
    private static final Map<String, Train> TRAINS = new ConcurrentHashMap<>();
    private static final List<Booking> BOOKINGS = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        seedTrains();

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.setExecutor(Executors.newFixedThreadPool(8));

        server.createContext("/health", App::handleHealth);
        server.createContext("/trains", App::handleTrains);
        server.createContext("/bookings", App::handleBookings);

        server.start();
        System.out.println("Train Booking backend running on http://localhost:8080");
        System.out.println("Endpoints:");
        System.out.println("GET  /health");
        System.out.println("GET  /trains");
        System.out.println("GET  /bookings");
        System.out.println("POST /bookings");
    }

    private static void seedTrains() {
        TRAINS.put("T101", new Train("T101", "Chennai", "Bangalore", 120));
        TRAINS.put("T202", new Train("T202", "Delhi", "Jaipur", 80));
        TRAINS.put("T303", new Train("T303", "Mumbai", "Pune", 150));
    }

    private static void handleHealth(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            writeJson(exchange, 405, new ErrorResponse("Method not allowed"));
            return;
        }
        writeJson(exchange, 200, Map.of("status", "UP"));
    }

    private static void handleTrains(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            writeJson(exchange, 405, new ErrorResponse("Method not allowed"));
            return;
        }

        List<Train> trainList = new ArrayList<>(TRAINS.values());
        writeJson(exchange, 200, trainList);
    }

    private static void handleBookings(HttpExchange exchange) throws IOException {
        String method = exchange.getRequestMethod();

        if ("GET".equalsIgnoreCase(method)) {
            synchronized (BOOKINGS) {
                writeJson(exchange, 200, BOOKINGS);
            }
            return;
        }

        if (!"POST".equalsIgnoreCase(method)) {
            writeJson(exchange, 405, new ErrorResponse("Method not allowed"));
            return;
        }

        String body = readRequestBody(exchange);
        BookingRequest request;

        try {
            request = GSON.fromJson(body, BookingRequest.class);
        } catch (JsonSyntaxException ex) {
            writeJson(exchange, 400, new ErrorResponse("Invalid JSON payload"));
            return;
        }

        if (request == null || isBlank(request.trainId) || isBlank(request.passengerName) || request.seats <= 0) {
            writeJson(exchange, 400, new ErrorResponse("trainId, passengerName and seats (>0) are required"));
            return;
        }

        Train train = TRAINS.get(request.trainId);
        if (train == null) {
            writeJson(exchange, 404, new ErrorResponse("Train not found: " + request.trainId));
            return;
        }

        Booking booking;
        synchronized (train) {
            if (train.availableSeats < request.seats) {
                writeJson(exchange, 409, new ErrorResponse("Not enough seats available"));
                return;
            }

            train.availableSeats -= request.seats;
            booking = new Booking(
                    UUID.randomUUID().toString(),
                    request.trainId,
                    request.passengerName,
                    request.seats,
                    LocalDateTime.now().toString()
            );
        }

        synchronized (BOOKINGS) {
            BOOKINGS.add(booking);
        }

        writeJson(exchange, 201, booking);
    }

    private static String readRequestBody(HttpExchange exchange) throws IOException {
        try (InputStream in = exchange.getRequestBody()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void writeJson(HttpExchange exchange, int statusCode, Object payload) throws IOException {
        String json = GSON.toJson(payload);
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);

        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(statusCode, bytes.length);

        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    static class Train {
        String id;
        String source;
        String destination;
        int availableSeats;

        Train(String id, String source, String destination, int availableSeats) {
            this.id = id;
            this.source = source;
            this.destination = destination;
            this.availableSeats = availableSeats;
        }
    }

    static class BookingRequest {
        String trainId;
        String passengerName;
        int seats;
    }

    static class Booking {
        String bookingId;
        String trainId;
        String passengerName;
        int seats;
        String bookedAt;

        Booking(String bookingId, String trainId, String passengerName, int seats, String bookedAt) {
            this.bookingId = bookingId;
            this.trainId = trainId;
            this.passengerName = passengerName;
            this.seats = seats;
            this.bookedAt = bookedAt;
        }
    }

    static class ErrorResponse {
        String error;

        ErrorResponse(String error) {
            this.error = error;
        }
    }
}
