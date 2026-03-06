import java.util.*;

public class CollectionsFramework {
    public static void main(String[] args){
    // Array
        ArrayList<String>list = new ArrayList<>();
        list.add("Bimal");
        list.add("Java");
        System.out.println(list.get(0));
    //LinkedList
        LinkedList<Integer> list2 = new LinkedList<>();

    //Set Interface
        HashSet<Integer>set = new HashSet<>();
        set.add(10);
        set.add(20);
        set.add(10);
        System.out.println(set);


    // HashMap
        HashMap<String,Integer>map = new HashMap<>();
        map.put("Bimal",23);
        map.put("Rahul",25);

        System.out.println(map.get("Bimal"));


    }
}
