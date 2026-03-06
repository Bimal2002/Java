package repository;

import models.Train;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class TrainRepository {
    private final List<Train> trains = new ArrayList<>();

    public TrainRepository() {
        seedTrains();
    }

    public List<Train> findAll() {
        return trains;
    }

    public Optional<Train> findById(String id) {
        return trains.stream().filter(train -> train.getId().equalsIgnoreCase(id)).findFirst();
    }

    public List<Train> search(String source, String destination) {
        String src = source.toLowerCase();
        String dst = destination.toLowerCase();
        return trains.stream()
                .filter(train -> train.getSource().toLowerCase().contains(src))
                .filter(train -> train.getDestination().toLowerCase().contains(dst))
                .collect(Collectors.toList());
    }

    private void seedTrains() {
        trains.add(new Train("T101", "Rajdhani Express", "Delhi", "Mumbai", 500, 120, 1550.0));
        trains.add(new Train("T202", "Shatabdi Express", "Bengaluru", "Chennai", 300, 80, 950.0));
        trains.add(new Train("T303", "Duronto Express", "Kolkata", "Delhi", 450, 60, 1320.0));
        trains.add(new Train("T404", "Intercity Express", "Hyderabad", "Pune", 350, 150, 880.0));
    }
}
