package models;

public class Train {
    private String id;
    private String name;
    private String source;
    private String destination;
    private int totalSeats;
    private int availableSeats;
    private double farePerSeat;

    public Train() {
    }

    public Train(String id, String name, String source, String destination, int totalSeats, int availableSeats, double farePerSeat) {
        this.id = id;
        this.name = name;
        this.source = source;
        this.destination = destination;
        this.totalSeats = totalSeats;
        this.availableSeats = availableSeats;
        this.farePerSeat = farePerSeat;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getSource() {
        return source;
    }

    public String getDestination() {
        return destination;
    }

    public int getTotalSeats() {
        return totalSeats;
    }

    public int getAvailableSeats() {
        return availableSeats;
    }

    public void setAvailableSeats(int availableSeats) {
        this.availableSeats = availableSeats;
    }

    public double getFarePerSeat() {
        return farePerSeat;
    }
}
