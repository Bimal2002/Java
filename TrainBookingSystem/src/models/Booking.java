package models;

public class Booking {
    private String id;
    private String userId;
    private String trainId;
    private String trainName;
    private String passengerName;
    private int seatsBooked;
    private double totalFare;
    private String bookingTime;

    public Booking() {
    }

    public Booking(String id, String userId, String trainId, String trainName, String passengerName, int seatsBooked, double totalFare, String bookingTime) {
        this.id = id;
        this.userId = userId;
        this.trainId = trainId;
        this.trainName = trainName;
        this.passengerName = passengerName;
        this.seatsBooked = seatsBooked;
        this.totalFare = totalFare;
        this.bookingTime = bookingTime;
    }

    public String getId() {
        return id;
    }

    public String getUserId() {
        return userId;
    }

    public String getTrainId() {
        return trainId;
    }

    public String getTrainName() {
        return trainName;
    }

    public String getPassengerName() {
        return passengerName;
    }

    public int getSeatsBooked() {
        return seatsBooked;
    }

    public double getTotalFare() {
        return totalFare;
    }

    public String getBookingTime() {
        return bookingTime;
    }
}
